"""
TrainingConfig: All configuration for the self-supervised training loop.

Includes:
    - Full configuration dataclass with all fields from SKILL.md
    - Validation with clear error messages
    - Serialization: to_dict(), from_dict(), to_yaml(), from_yaml()
    - Compatibility check for config version mismatches

CRITICAL: All config dtypes ('bfloat16', 'float16') map to torch.dtype objects.
          Module inference mode: always use module.train(False).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Tuple, Dict, Any
import torch


# Current config version — increment when adding required fields
CONFIG_VERSION = '0.1.0'

# Mapping from string dtype names to torch.dtype objects
DTYPE_MAP: Dict[str, torch.dtype] = {
    'bfloat16': torch.bfloat16,
    'float16':  torch.float16,
    'float32':  torch.float32,
}


@dataclass
class TrainingConfig:
    """
    Complete configuration for a self-supervised training loop.

    AMP fields:
        amp_enabled:       Enable AMP mixed precision training.
        amp_dtype:         Precision type ('bfloat16' or 'float16'). bfloat16 recommended.
        grad_scaler_enabled: Enable GradScaler for inf/nan detection. Useful even with bfloat16.
        max_grad_norm:     Gradient clipping threshold (applied at true gradient scale after unscale_).

    Optimizer fields:
        lr:            Peak learning rate. ViT standard: 1e-4.
        weight_decay:  AdamW weight decay. ViT standard: 0.04.
        betas:         AdamW betas. (0.9, 0.95) — 0.95 beta2 reduces second-moment lag for ViTs.
        warmup_steps:  Linear warmup steps. Cosine decay begins after.
        lr_min:        Minimum learning rate at end of cosine decay.
        total_steps:   Total training steps (includes warmup).

    EMA fields:
        ema_tau_base:   EMA tau at step 0 (fast tracking). Typical: 0.996.
        ema_tau_final:  EMA tau at total_steps (near-frozen). Typical: 0.9999.

    Checkpointing fields:
        checkpoint_dir:   Directory to save/load checkpoints.
        checkpoint_every: Save frequency in steps.
        auto_resume:      If True, scan checkpoint_dir on startup and resume latest.
        keep_checkpoints: Number of most-recent checkpoints to retain.

    Logging fields:
        wandb_project:    W&B project name.
        wandb_entity:     W&B entity (team or user name). None for personal account.
        log_every:        Log metrics every N steps (1 = every step).
        sample_grid_every: Log prediction grid every N steps.
        grid_nrow:        Number of images per row in prediction grid.

    DDP fields:
        backend:          Distributed backend. 'nccl' for GPU, 'gloo' for CPU.
        sync_batchnorm:   Convert BatchNorm to SyncBatchNorm. Set False for ViT.
    """

    # AMP
    amp_enabled: bool = True
    amp_dtype: str = 'bfloat16'
    grad_scaler_enabled: bool = True
    max_grad_norm: float = 1.0

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.04
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 10_000
    lr_min: float = 1e-6
    total_steps: int = 100_000

    # EMA
    ema_tau_base: float = 0.996
    ema_tau_final: float = 0.9999

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    checkpoint_every: int = 5_000
    auto_resume: bool = True
    keep_checkpoints: int = 3

    # Logging
    wandb_project: str = 'ssl-training'
    wandb_entity: Optional[str] = None
    log_every: int = 1
    sample_grid_every: int = 1_000
    grid_nrow: int = 8

    # DDP
    backend: str = 'nccl'
    sync_batchnorm: bool = True

    # Internal versioning
    _version: str = field(default=CONFIG_VERSION, repr=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate all config fields. Raises ValueError with a clear message on error.

        Call after creating or modifying the config, before starting training.
        """
        errors = []

        # AMP dtype validation
        if self.amp_dtype not in DTYPE_MAP:
            errors.append(
                f"amp_dtype must be one of {list(DTYPE_MAP.keys())}, got '{self.amp_dtype}'"
            )

        # Learning rate
        if self.lr <= 0:
            errors.append(f"lr must be positive, got {self.lr}")

        if self.lr_min < 0:
            errors.append(f"lr_min must be non-negative, got {self.lr_min}")

        if self.lr_min >= self.lr:
            errors.append(
                f"lr_min ({self.lr_min}) must be less than lr ({self.lr})"
            )

        # Gradient clipping
        if self.max_grad_norm <= 0:
            errors.append(f"max_grad_norm must be positive, got {self.max_grad_norm}")

        # Weight decay
        if self.weight_decay < 0:
            errors.append(f"weight_decay must be non-negative, got {self.weight_decay}")

        # Betas
        if not (0.0 < self.betas[0] < 1.0):
            errors.append(f"betas[0] must be in (0, 1), got {self.betas[0]}")
        if not (0.0 < self.betas[1] < 1.0):
            errors.append(f"betas[1] must be in (0, 1), got {self.betas[1]}")

        # Schedule
        if self.warmup_steps < 0:
            errors.append(f"warmup_steps must be non-negative, got {self.warmup_steps}")

        if self.total_steps <= 0:
            errors.append(f"total_steps must be positive, got {self.total_steps}")

        if self.warmup_steps >= self.total_steps:
            errors.append(
                f"warmup_steps ({self.warmup_steps}) must be less than "
                f"total_steps ({self.total_steps})"
            )

        # EMA
        if not (0 < self.ema_tau_base < 1):
            errors.append(f"ema_tau_base must be in (0, 1), got {self.ema_tau_base}")

        if not (0 < self.ema_tau_final <= 1):
            errors.append(f"ema_tau_final must be in (0, 1], got {self.ema_tau_final}")

        if self.ema_tau_base >= self.ema_tau_final:
            errors.append(
                f"ema_tau_base ({self.ema_tau_base}) must be less than "
                f"ema_tau_final ({self.ema_tau_final})"
            )

        # Checkpointing
        if self.checkpoint_every <= 0:
            errors.append(f"checkpoint_every must be positive, got {self.checkpoint_every}")

        if self.keep_checkpoints < 1:
            errors.append(f"keep_checkpoints must be at least 1, got {self.keep_checkpoints}")

        # Logging
        if self.log_every <= 0:
            errors.append(f"log_every must be positive, got {self.log_every}")

        if self.sample_grid_every <= 0:
            errors.append(f"sample_grid_every must be positive, got {self.sample_grid_every}")

        if self.grid_nrow <= 0:
            errors.append(f"grid_nrow must be positive, got {self.grid_nrow}")

        # DDP backend
        valid_backends = {'nccl', 'gloo', 'mpi'}
        if self.backend not in valid_backends:
            errors.append(f"backend must be one of {valid_backends}, got '{self.backend}'")

        if errors:
            raise ValueError(
                "TrainingConfig validation failed with the following errors:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # dtype property
    # ------------------------------------------------------------------

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the torch.dtype corresponding to amp_dtype string."""
        return DTYPE_MAP.get(self.amp_dtype, torch.bfloat16)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize config to a plain Python dict.

        betas tuple is converted to list for JSON compatibility.
        Internal _version field is included for compatibility checking.
        """
        d = asdict(self)
        # Convert tuple to list for JSON compatibility
        d['betas'] = list(d['betas'])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """
        Deserialize config from a dict (e.g., loaded from JSON/YAML).

        Handles:
        - betas as list -> tuple conversion
        - Missing fields (uses dataclass defaults)
        - Extra unknown fields (ignored with warning)
        """
        # Convert betas list back to tuple
        if 'betas' in data and isinstance(data['betas'], (list, tuple)):
            data = dict(data)
            data['betas'] = tuple(data['betas'])

        # Get valid field names
        valid_fields = {f.name for f in fields(cls)}

        # Filter out unknown fields (warn but don't fail)
        unknown = set(data.keys()) - valid_fields
        if unknown:
            print(f"[TrainingConfig] Warning: Unknown config fields ignored: {unknown}")

        known_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**known_data)

    def to_json(self) -> str:
        """Serialize config to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'TrainingConfig':
        """Deserialize config from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        """
        Serialize config to YAML format string.

        Note: Requires PyYAML (yaml package). Falls back to JSON-like format if unavailable.
        """
        try:
            import yaml
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=True)
        except ImportError:
            # Fallback: indented key-value format (not strict YAML)
            lines = []
            for key, value in sorted(self.to_dict().items()):
                lines.append(f"{key}: {repr(value)}")
            return "\n".join(lines)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'TrainingConfig':
        """
        Deserialize config from a YAML string.

        Requires PyYAML (yaml package).
        """
        try:
            import yaml
            data = yaml.safe_load(yaml_str)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML deserialization. "
                "Install with: pip install pyyaml"
            )
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------

    def check_compatibility(self, other: 'TrainingConfig') -> Tuple[bool, list]:
        """
        Check if two configs are compatible for resuming training.

        Returns:
            (compatible, list_of_incompatibilities)

        Incompatibilities that break resume:
            - Different total_steps (changes scheduler curve)
            - Different warmup_steps (changes scheduler)
            - Different ema_tau_base or ema_tau_final (changes EMA curve)
            - Different amp_dtype (changes gradient precision)
        """
        critical_fields = [
            'total_steps', 'warmup_steps', 'lr_min', 'lr',
            'ema_tau_base', 'ema_tau_final', 'amp_dtype',
        ]
        incompatibilities = []

        for fname in critical_fields:
            val_self = getattr(self, fname)
            val_other = getattr(other, fname)
            if val_self != val_other:
                incompatibilities.append(
                    f"{fname}: saved={val_other!r}, current={val_self!r}"
                )

        return len(incompatibilities) == 0, incompatibilities

    def __post_init__(self):
        """Convert betas list/tuple to tuple if needed after construction."""
        if isinstance(self.betas, list):
            object.__setattr__(self, 'betas', tuple(self.betas))

    def __repr__(self) -> str:
        fields_str = ", ".join(
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if not f.name.startswith('_')
        )
        return f"TrainingConfig({fields_str})"


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TrainingConfig Self-Tests")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Test 1: Default config is valid
    # ----------------------------------------------------------------
    print("\nTest 1: Default config passes validation...")

    cfg = TrainingConfig()
    cfg.validate()  # Should not raise
    print(f"  PASS: Default config valid (lr={cfg.lr}, total_steps={cfg.total_steps})")

    # ----------------------------------------------------------------
    # Test 2: Bad amp_dtype raises ValueError
    # ----------------------------------------------------------------
    print("\nTest 2: Bad amp_dtype raises ValueError...")

    cfg_bad = TrainingConfig(amp_dtype='fp16')  # Not a valid string
    try:
        cfg_bad.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'amp_dtype' in str(e), f"Error should mention amp_dtype: {e}"
        print(f"  PASS: ValueError raised: {str(e)[:80]}...")

    # ----------------------------------------------------------------
    # Test 3: warmup_steps >= total_steps raises ValueError
    # ----------------------------------------------------------------
    print("\nTest 3: warmup_steps >= total_steps raises ValueError...")

    cfg_bad2 = TrainingConfig(warmup_steps=100_000, total_steps=100_000)
    try:
        cfg_bad2.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'warmup_steps' in str(e), f"Error should mention warmup_steps: {e}"
        print(f"  PASS: ValueError raised: {str(e)[:80]}...")

    # ----------------------------------------------------------------
    # Test 4: tau_base >= tau_final raises ValueError
    # ----------------------------------------------------------------
    print("\nTest 4: ema_tau_base >= ema_tau_final raises ValueError...")

    cfg_bad3 = TrainingConfig(ema_tau_base=0.9999, ema_tau_final=0.996)
    try:
        cfg_bad3.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'ema_tau_base' in str(e), f"Error should mention ema_tau_base: {e}"
        print(f"  PASS: ValueError raised: {str(e)[:80]}...")

    # ----------------------------------------------------------------
    # Test 5: lr <= 0 raises ValueError
    # ----------------------------------------------------------------
    print("\nTest 5: lr <= 0 raises ValueError...")

    cfg_bad4 = TrainingConfig(lr=-1e-4)
    try:
        cfg_bad4.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'lr' in str(e), f"Error should mention lr: {e}"
        print(f"  PASS: ValueError raised: {str(e)[:80]}...")

    # ----------------------------------------------------------------
    # Test 6: to_dict / from_dict round-trip
    # ----------------------------------------------------------------
    print("\nTest 6: to_dict / from_dict round-trip...")

    cfg_orig = TrainingConfig(lr=5e-4, warmup_steps=500, total_steps=5_000)
    d = cfg_orig.to_dict()

    assert isinstance(d, dict), "to_dict() must return a dict"
    assert 'lr' in d, "to_dict() must include 'lr'"
    assert isinstance(d['betas'], list), "betas must be list in dict (JSON-compatible)"

    cfg_restored = TrainingConfig.from_dict(d)
    assert cfg_restored.lr == cfg_orig.lr, f"lr mismatch: {cfg_restored.lr} != {cfg_orig.lr}"
    assert cfg_restored.betas == cfg_orig.betas, (
        f"betas mismatch: {cfg_restored.betas} != {cfg_orig.betas}"
    )
    assert cfg_restored.warmup_steps == cfg_orig.warmup_steps, "warmup_steps mismatch"
    print(f"  PASS: Round-trip successful (lr={cfg_restored.lr}, betas={cfg_restored.betas})")

    # ----------------------------------------------------------------
    # Test 7: JSON round-trip
    # ----------------------------------------------------------------
    print("\nTest 7: to_json / from_json round-trip...")

    cfg_json = TrainingConfig(wandb_project='test-project', log_every=10)
    json_str = cfg_json.to_json()

    assert isinstance(json_str, str), "to_json() must return a string"
    # Verify it's valid JSON
    parsed = json.loads(json_str)
    assert parsed['wandb_project'] == 'test-project', "wandb_project mismatch in JSON"

    cfg_from_json = TrainingConfig.from_json(json_str)
    assert cfg_from_json.wandb_project == 'test-project', "wandb_project mismatch after from_json"
    assert cfg_from_json.log_every == 10, "log_every mismatch after from_json"
    print(f"  PASS: JSON round-trip successful")

    # ----------------------------------------------------------------
    # Test 8: torch_dtype property
    # ----------------------------------------------------------------
    print("\nTest 8: torch_dtype property returns correct dtype...")

    cfg_bf16 = TrainingConfig(amp_dtype='bfloat16')
    cfg_f16 = TrainingConfig(amp_dtype='float16')

    assert cfg_bf16.torch_dtype == torch.bfloat16, (
        f"bfloat16 config should return torch.bfloat16, got {cfg_bf16.torch_dtype}"
    )
    assert cfg_f16.torch_dtype == torch.float16, (
        f"float16 config should return torch.float16, got {cfg_f16.torch_dtype}"
    )
    print(f"  PASS: bfloat16 -> {cfg_bf16.torch_dtype}, float16 -> {cfg_f16.torch_dtype}")

    # ----------------------------------------------------------------
    # Test 9: compatibility check
    # ----------------------------------------------------------------
    print("\nTest 9: check_compatibility detects incompatible configs...")

    cfg_a = TrainingConfig(total_steps=100_000, lr=1e-4)
    cfg_b = TrainingConfig(total_steps=200_000, lr=1e-4)

    compatible, issues = cfg_a.check_compatibility(cfg_b)
    assert not compatible, "Configs with different total_steps should be incompatible"
    assert any('total_steps' in issue for issue in issues), (
        f"Incompatibility should mention total_steps, got: {issues}"
    )
    print(f"  PASS: Incompatibility detected: {issues[0]}")

    # Same config is compatible with itself
    compatible_self, issues_self = cfg_a.check_compatibility(cfg_a)
    assert compatible_self, "Config should be compatible with itself"
    print(f"  PASS: Self-compatible, no issues")

    # ----------------------------------------------------------------
    # Test 10: from_dict ignores unknown fields with warning
    # ----------------------------------------------------------------
    print("\nTest 10: from_dict ignores unknown fields gracefully...")

    d_with_unknown = {
        'lr': 1e-3,
        'total_steps': 5000,
        'warmup_steps': 500,
        'unknown_future_field': 'some_value',
        'another_unknown': 42,
    }
    cfg_from_unknown = TrainingConfig.from_dict(d_with_unknown)
    assert cfg_from_unknown.lr == 1e-3, "lr should be set from dict"
    # Verify unknown fields didn't cause errors
    print("  PASS: Unknown fields ignored, config created successfully")

    # ----------------------------------------------------------------
    # Test 11: checkpoint_every <= 0 raises ValueError
    # ----------------------------------------------------------------
    print("\nTest 11: checkpoint_every <= 0 raises ValueError...")

    cfg_bad5 = TrainingConfig(checkpoint_every=0)
    try:
        cfg_bad5.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'checkpoint_every' in str(e), f"Error should mention checkpoint_every: {e}"
        print(f"  PASS: ValueError raised: {str(e)[:80]}...")

    print("\n" + "=" * 60)
    print("All TrainingConfig self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
