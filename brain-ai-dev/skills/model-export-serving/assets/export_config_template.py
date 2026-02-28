"""
Export configuration dataclasses: ExportConfig, QuantizationConfig,
PruningConfig, ServingConfig, and unified PipelineConfig.

All configs include validation, serialization, and preset factories.

torch + standard lib only.
"""

import copy
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# ExportConfig
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """Configuration for model export (ONNX / TorchScript)."""
    format: str = "torchscript"  # onnx | torchscript | both
    opset_version: int = 17
    dynamic_axes: bool = True
    snn_mode: str = "stateless"  # stateless | stateful
    validate_export: bool = True
    tolerance: float = 1e-4
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])
    torchscript_method: str = "trace"  # trace | script | hybrid
    optimize_torchscript: bool = True
    freeze_torchscript: bool = False
    check_trace: bool = True
    verbose: bool = False

    def validate(self) -> List[str]:
        """Validate configuration, returning list of issues."""
        issues: List[str] = []
        if self.format not in ("onnx", "torchscript", "both"):
            issues.append(f"Invalid format: {self.format}")
        if self.opset_version < 9:
            issues.append(f"Opset version {self.opset_version} too low (min 9)")
        if self.opset_version > 21:
            issues.append(f"Opset version {self.opset_version} may not be supported yet")
        if self.snn_mode not in ("stateless", "stateful"):
            issues.append(f"Invalid snn_mode: {self.snn_mode}")
        if self.tolerance <= 0:
            issues.append(f"Tolerance must be positive, got {self.tolerance}")
        if self.torchscript_method not in ("trace", "script", "hybrid"):
            issues.append(f"Invalid torchscript_method: {self.torchscript_method}")
        if not self.input_names:
            issues.append("input_names must not be empty")
        if not self.output_names:
            issues.append("output_names must not be empty")
        return issues

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExportConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def onnx_default(cls) -> "ExportConfig":
        return cls(format="onnx", opset_version=17)

    @classmethod
    def torchscript_trace(cls) -> "ExportConfig":
        return cls(format="torchscript", torchscript_method="trace")

    @classmethod
    def torchscript_script(cls) -> "ExportConfig":
        return cls(format="torchscript", torchscript_method="script")

    @classmethod
    def production(cls) -> "ExportConfig":
        return cls(
            format="both",
            opset_version=17,
            validate_export=True,
            tolerance=1e-4,
            optimize_torchscript=True,
            freeze_torchscript=True,
        )


# ---------------------------------------------------------------------------
# QuantizationConfig
# ---------------------------------------------------------------------------

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: str = "dynamic"  # dynamic | static | qat | fp16
    dtype: str = "int8"  # int8 | fp16 | bf16
    backend: str = "x86"  # x86 | qnnpack | fbgemm
    calibration_samples: int = 1000
    accuracy_threshold: float = 0.02
    per_channel: bool = True
    use_histogram_observer: bool = False
    skip_modules: List[str] = field(default_factory=list)
    sensitive_modules: List[str] = field(default_factory=lambda: [
        "snn_core", "neuromodulator", "workspace.attention", "blend_gate",
    ])
    qat_epochs: int = 5
    qat_lr_scale: float = 0.1
    verbose: bool = False

    def validate(self) -> List[str]:
        issues: List[str] = []
        if self.method not in ("dynamic", "static", "qat", "fp16"):
            issues.append(f"Invalid method: {self.method}")
        if self.dtype not in ("int8", "fp16", "bf16"):
            issues.append(f"Invalid dtype: {self.dtype}")
        if self.backend not in ("x86", "qnnpack", "fbgemm"):
            issues.append(f"Invalid backend: {self.backend}")
        if self.calibration_samples < 1:
            issues.append(f"calibration_samples must be >= 1")
        if self.accuracy_threshold <= 0 or self.accuracy_threshold >= 1.0:
            issues.append(f"accuracy_threshold should be in (0, 1), got {self.accuracy_threshold}")
        if self.qat_epochs < 1:
            issues.append(f"qat_epochs must be >= 1")
        if self.qat_lr_scale <= 0:
            issues.append(f"qat_lr_scale must be positive")
        return issues

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def dynamic_int8(cls) -> "QuantizationConfig":
        return cls(method="dynamic", dtype="int8")

    @classmethod
    def static_int8(cls) -> "QuantizationConfig":
        return cls(method="static", dtype="int8", use_histogram_observer=True)

    @classmethod
    def fp16_gpu(cls) -> "QuantizationConfig":
        return cls(method="fp16", dtype="fp16")

    @classmethod
    def qat_int8(cls) -> "QuantizationConfig":
        return cls(method="qat", dtype="int8", qat_epochs=5)

    @classmethod
    def conservative(cls) -> "QuantizationConfig":
        """Conservative quantization: skip all sensitive modules."""
        return cls(
            method="dynamic",
            dtype="int8",
            accuracy_threshold=0.01,
            skip_modules=[
                "snn_core", "neuromodulator", "workspace.attention",
                "blend_gate", "decision_heads.final",
            ],
        )


# ---------------------------------------------------------------------------
# PruningConfig
# ---------------------------------------------------------------------------

@dataclass
class PruningConfig:
    """Configuration for model pruning."""
    method: str = "unstructured"  # unstructured | structured | global
    sparsity: float = 0.3
    norm: int = 1  # L1 or L2
    structured_dim: int = 0  # 0 = output channels
    target_layers: List[str] = field(default_factory=list)
    skip_layers: List[str] = field(default_factory=list)
    iterative_rounds: int = 1
    make_permanent: bool = True
    retrain_after_prune: bool = False
    retrain_epochs: int = 3
    retrain_lr_scale: float = 0.1
    verbose: bool = False

    def validate(self) -> List[str]:
        issues: List[str] = []
        if self.method not in ("unstructured", "structured", "global"):
            issues.append(f"Invalid method: {self.method}")
        if self.sparsity < 0 or self.sparsity >= 1.0:
            issues.append(f"Sparsity must be in [0, 1), got {self.sparsity}")
        if self.norm not in (1, 2):
            issues.append(f"Norm must be 1 or 2, got {self.norm}")
        if self.iterative_rounds < 1:
            issues.append(f"iterative_rounds must be >= 1")
        if self.retrain_epochs < 1:
            issues.append(f"retrain_epochs must be >= 1")
        return issues

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PruningConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def light(cls) -> "PruningConfig":
        return cls(method="unstructured", sparsity=0.2)

    @classmethod
    def moderate(cls) -> "PruningConfig":
        return cls(method="unstructured", sparsity=0.5, iterative_rounds=3)

    @classmethod
    def aggressive(cls) -> "PruningConfig":
        return cls(
            method="global", sparsity=0.7,
            iterative_rounds=5, retrain_after_prune=True,
        )

    @classmethod
    def structured_channels(cls) -> "PruningConfig":
        return cls(method="structured", sparsity=0.3, structured_dim=0)


# ---------------------------------------------------------------------------
# ServingConfig
# ---------------------------------------------------------------------------

@dataclass
class ServingConfig:
    """Configuration for model serving infrastructure."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    num_workers: int = 1
    device: str = "auto"  # auto | cpu | cuda | cuda:0 etc.
    model_path: str = "model.pt"
    model_format: str = "torchscript"  # torchscript | onnx
    warmup_iterations: int = 5
    request_timeout_seconds: float = 30.0
    max_queue_depth: int = 1000
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    cors_origins: List[str] = field(default_factory=list)
    log_level: str = "info"
    verbose: bool = False

    def validate(self) -> List[str]:
        issues: List[str] = []
        if self.port < 1 or self.port > 65535:
            issues.append(f"Port must be 1-65535, got {self.port}")
        if self.max_batch_size < 1:
            issues.append(f"max_batch_size must be >= 1")
        if self.batch_timeout_ms < 0:
            issues.append(f"batch_timeout_ms must be >= 0")
        if self.num_workers < 1:
            issues.append(f"num_workers must be >= 1")
        if self.request_timeout_seconds <= 0:
            issues.append(f"request_timeout_seconds must be positive")
        if self.max_queue_depth < 1:
            issues.append(f"max_queue_depth must be >= 1")
        if self.model_format not in ("torchscript", "onnx"):
            issues.append(f"Invalid model_format: {self.model_format}")
        if self.log_level not in ("debug", "info", "warning", "error"):
            issues.append(f"Invalid log_level: {self.log_level}")
        return issues

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ServingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def development(cls) -> "ServingConfig":
        return cls(
            host="127.0.0.1",
            port=8000,
            max_batch_size=4,
            num_workers=1,
            device="cpu",
            warmup_iterations=1,
            log_level="debug",
            verbose=True,
        )

    @classmethod
    def production_cpu(cls) -> "ServingConfig":
        return cls(
            host="0.0.0.0",
            port=8000,
            max_batch_size=32,
            num_workers=4,
            device="cpu",
            warmup_iterations=10,
            log_level="info",
        )

    @classmethod
    def production_gpu(cls) -> "ServingConfig":
        return cls(
            host="0.0.0.0",
            port=8000,
            max_batch_size=64,
            num_workers=1,
            device="cuda",
            warmup_iterations=20,
            log_level="info",
        )


# ---------------------------------------------------------------------------
# PipelineConfig (unified)
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Unified configuration for the full export-quantize-prune-serve pipeline."""
    export: ExportConfig = field(default_factory=ExportConfig)
    quantization: Optional[QuantizationConfig] = None
    pruning: Optional[PruningConfig] = None
    serving: ServingConfig = field(default_factory=ServingConfig)

    def validate(self) -> List[str]:
        issues: List[str] = []
        issues.extend([f"export: {i}" for i in self.export.validate()])
        if self.quantization is not None:
            issues.extend([f"quantization: {i}" for i in self.quantization.validate()])
        if self.pruning is not None:
            issues.extend([f"pruning: {i}" for i in self.pruning.validate()])
        issues.extend([f"serving: {i}" for i in self.serving.validate()])
        return issues

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "export": self.export.to_dict(),
            "serving": self.serving.to_dict(),
        }
        if self.quantization is not None:
            d["quantization"] = self.quantization.to_dict()
        if self.pruning is not None:
            d["pruning"] = self.pruning.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        export = ExportConfig.from_dict(d.get("export", {}))
        serving = ServingConfig.from_dict(d.get("serving", {}))
        quant = QuantizationConfig.from_dict(d["quantization"]) if "quantization" in d else None
        prune = PruningConfig.from_dict(d["pruning"]) if "pruning" in d else None
        return cls(export=export, quantization=quant, pruning=prune, serving=serving)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PipelineConfig":
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            return cls.from_json(f.read())

    @classmethod
    def minimal_dev(cls) -> "PipelineConfig":
        return cls(
            export=ExportConfig.torchscript_trace(),
            serving=ServingConfig.development(),
        )

    @classmethod
    def production_int8_cpu(cls) -> "PipelineConfig":
        return cls(
            export=ExportConfig.production(),
            quantization=QuantizationConfig.dynamic_int8(),
            pruning=PruningConfig.light(),
            serving=ServingConfig.production_cpu(),
        )

    @classmethod
    def production_fp16_gpu(cls) -> "PipelineConfig":
        return cls(
            export=ExportConfig.production(),
            quantization=QuantizationConfig.fp16_gpu(),
            serving=ServingConfig.production_gpu(),
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:  # noqa: C901
    """Run 30+ self-tests for config dataclasses."""
    passed = 0
    failed = 0

    def _ok(name: str, cond: bool) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    print("=" * 60)
    print("Export Config Self-Tests")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="config_test_")

    # --- ExportConfig ---
    ec = ExportConfig()
    _ok("T01 EC default format torchscript", ec.format == "torchscript")
    _ok("T02 EC default opset 17", ec.opset_version == 17)
    _ok("T03 EC validate no issues", len(ec.validate()) == 0)

    ec_bad = ExportConfig(format="invalid", opset_version=5, tolerance=-1)
    _ok("T04 EC validate catches bad format", len(ec_bad.validate()) > 0)

    ec_onnx = ExportConfig.onnx_default()
    _ok("T05 EC ONNX preset format", ec_onnx.format == "onnx")

    ec_ts = ExportConfig.torchscript_trace()
    _ok("T06 EC TS trace preset", ec_ts.torchscript_method == "trace")

    ec_prod = ExportConfig.production()
    _ok("T07 EC production format both", ec_prod.format == "both")

    ec_dict = ec.to_dict()
    _ok("T08 EC to_dict", isinstance(ec_dict, dict))

    ec_from = ExportConfig.from_dict(ec_dict)
    _ok("T09 EC from_dict round-trip", ec_from.format == ec.format)

    # --- QuantizationConfig ---
    qc = QuantizationConfig()
    _ok("T10 QC default method dynamic", qc.method == "dynamic")
    _ok("T11 QC validate no issues", len(qc.validate()) == 0)

    qc_bad = QuantizationConfig(method="invalid", calibration_samples=0)
    _ok("T12 QC validate catches issues", len(qc_bad.validate()) > 0)

    qc_dyn = QuantizationConfig.dynamic_int8()
    _ok("T13 QC dynamic preset", qc_dyn.method == "dynamic")

    qc_static = QuantizationConfig.static_int8()
    _ok("T14 QC static preset histogram", qc_static.use_histogram_observer)

    qc_fp16 = QuantizationConfig.fp16_gpu()
    _ok("T15 QC fp16 preset", qc_fp16.method == "fp16")

    qc_qat = QuantizationConfig.qat_int8()
    _ok("T16 QC qat preset epochs", qc_qat.qat_epochs == 5)

    qc_cons = QuantizationConfig.conservative()
    _ok("T17 QC conservative skip modules", len(qc_cons.skip_modules) > 0)

    qc_dict = qc.to_dict()
    qc_from = QuantizationConfig.from_dict(qc_dict)
    _ok("T18 QC round-trip", qc_from.method == qc.method)

    # --- PruningConfig ---
    pc = PruningConfig()
    _ok("T19 PC default method unstructured", pc.method == "unstructured")
    _ok("T20 PC default sparsity 0.3", pc.sparsity == 0.3)
    _ok("T21 PC validate no issues", len(pc.validate()) == 0)

    pc_bad = PruningConfig(method="invalid", sparsity=1.5, norm=3)
    _ok("T22 PC validate catches issues", len(pc_bad.validate()) >= 2)

    pc_light = PruningConfig.light()
    _ok("T23 PC light sparsity 0.2", pc_light.sparsity == 0.2)

    pc_mod = PruningConfig.moderate()
    _ok("T24 PC moderate rounds 3", pc_mod.iterative_rounds == 3)

    pc_agg = PruningConfig.aggressive()
    _ok("T25 PC aggressive sparsity 0.7", pc_agg.sparsity == 0.7)

    pc_struct = PruningConfig.structured_channels()
    _ok("T26 PC structured dim 0", pc_struct.structured_dim == 0)

    # --- ServingConfig ---
    sc = ServingConfig()
    _ok("T27 SC default port 8000", sc.port == 8000)
    _ok("T28 SC validate no issues", len(sc.validate()) == 0)

    sc_bad = ServingConfig(port=0, max_batch_size=0, model_format="invalid")
    _ok("T29 SC validate catches issues", len(sc_bad.validate()) >= 2)

    sc_dev = ServingConfig.development()
    _ok("T30 SC dev localhost", sc_dev.host == "127.0.0.1")

    sc_cpu = ServingConfig.production_cpu()
    _ok("T31 SC prod CPU workers 4", sc_cpu.num_workers == 4)

    sc_gpu = ServingConfig.production_gpu()
    _ok("T32 SC prod GPU device", sc_gpu.device == "cuda")

    # --- PipelineConfig ---
    pipe = PipelineConfig()
    _ok("T33 Pipeline default valid", len(pipe.validate()) == 0)

    pipe_full = PipelineConfig(
        export=ExportConfig.production(),
        quantization=QuantizationConfig.dynamic_int8(),
        pruning=PruningConfig.light(),
        serving=ServingConfig.production_cpu(),
    )
    _ok("T34 Full pipeline valid", len(pipe_full.validate()) == 0)

    pipe_dict = pipe_full.to_dict()
    _ok("T35 Pipeline to_dict has export", "export" in pipe_dict)
    _ok("T36 Pipeline to_dict has quantization", "quantization" in pipe_dict)

    pipe_from = PipelineConfig.from_dict(pipe_dict)
    _ok("T37 Pipeline from_dict round-trip", pipe_from.export.format == pipe_full.export.format)

    # JSON round-trip
    pipe_json = pipe_full.to_json()
    pipe_from_json = PipelineConfig.from_json(pipe_json)
    _ok("T38 Pipeline JSON round-trip", pipe_from_json.serving.port == 8000)

    # Save/load
    save_path = os.path.join(tmpdir, "pipeline.json")
    pipe_full.save(save_path)
    pipe_loaded = PipelineConfig.load(save_path)
    _ok("T39 Pipeline save/load", pipe_loaded.export.opset_version == 17)

    # Presets
    pipe_dev = PipelineConfig.minimal_dev()
    _ok("T40 Pipeline minimal dev", pipe_dev.quantization is None)

    pipe_int8 = PipelineConfig.production_int8_cpu()
    _ok("T41 Pipeline prod INT8", pipe_int8.quantization is not None)
    _ok("T42 Pipeline prod INT8 has pruning", pipe_int8.pruning is not None)

    pipe_fp16 = PipelineConfig.production_fp16_gpu()
    _ok("T43 Pipeline prod FP16 GPU", pipe_fp16.serving.device == "cuda")

    # Validate catches nested issues
    pipe_broken = PipelineConfig(
        export=ExportConfig(format="invalid"),
        serving=ServingConfig(port=-1),
    )
    issues = pipe_broken.validate()
    _ok("T44 Nested validation catches issues", len(issues) >= 2)

    # Copy/deepcopy
    pipe_copy = copy.deepcopy(pipe_full)
    pipe_copy.export.opset_version = 14
    _ok("T45 Deepcopy independent", pipe_full.export.opset_version == 17)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Temp dir: {tmpdir}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
