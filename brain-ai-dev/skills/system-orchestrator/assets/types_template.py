"""
brain_ai/types.py — Typed contracts for the BrainAI pipeline.

All module boundaries use these dataclasses to enforce shape, dtype, and device
invariants. The orchestrator (system.py) asserts contracts cheaply at runtime.

Usage:
    from brain_ai.types import (
        ModalityBatch, EncoderOutput, WorkspaceOutput, HTMOutput,
        ReasoningOutput, DecisionOutput, SystemOutput, SystemDetails,
        BrainAIState,
    )
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Canonical Input
# ---------------------------------------------------------------------------

@dataclass
class ModalityBatch:
    """Normalized input batch consumed by all downstream modules.

    The orchestrator converts raw user inputs (tensors, strings, file paths,
    dicts with missing modalities) into this canonical form.
    """
    # Core modalities (None if not provided)
    vision: Optional[Tensor] = None           # (B,C,H,W) or (B,T,C,H,W)
    vision_mask: Optional[Tensor] = None      # (B,) or (B,T) boolean
    text: Optional[Tensor] = None             # (B,L) int64 token ids
    attention_mask: Optional[Tensor] = None   # (B,L) boolean
    audio: Optional[Tensor] = None            # (B,S) raw waveform or (B,T,F) features
    audio_mask: Optional[Tensor] = None       # (B,) or (B,T)
    sensors: Optional[Tensor] = None          # (B,T,D)
    sensor_mask: Optional[Tensor] = None      # (B,T)

    # Control / RL signals (for active inference / neuromodulation)
    reward: Optional[Tensor] = None           # (B,) or (B,1)
    done: Optional[Tensor] = None             # (B,) boolean
    action: Optional[Tensor] = None           # (B,A) previous action

    # Engram (token-level)
    token_ids: Optional[Tensor] = None        # (B,L) for engram encoder

    # Metadata
    modalities_present: List[str] = field(default_factory=list)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    compute_dtype: torch.dtype = torch.float32


# ---------------------------------------------------------------------------
# Module Output Contracts
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    """Output from any modality encoder.

    Invariants:
        feats.shape == (B, T, D) — always 3D, even if T=1
        mask.shape == (B, T)
        salience.shape == (B, T) or (B, 1)
        feats.shape[-1] == config.encoder.output_dim
    """
    modality: str
    feats: Tensor                             # (B, T, D)
    mask: Tensor                              # (B, T) boolean
    salience: Tensor                          # (B, T) or (B, 1)
    aux: Optional[Dict[str, Tensor]] = None


@dataclass
class WorkspaceOutput:
    """Output from Global Workspace competition.

    Invariants:
        slots.shape == (B, K, D) where K <= capacity_limit
        D == config.workspace.workspace_dim
        winners sorted by descending score
    """
    slots: Tensor                             # (B, K, D)
    slot_mask: Tensor                         # (B, K) boolean
    winners: Tensor                           # (B, K) indices
    winner_scores: Tensor                     # (B, K) competition scores
    attn: Optional[Tensor] = None             # (B, H, K, T_total) attention maps
    broadcast: Optional[Tensor] = None        # (B, D) broadcast signal
    wm_state: Optional[Any] = None            # Working memory updated state
    modality_contributions: Optional[Dict[str, Tensor]] = None


@dataclass
class HTMOutput:
    """Output from HTM temporal layer.

    Invariants:
        anomaly_score.shape == (B,) and values in [0, 1]
        prediction.shape[-1] == workspace_dim (if present)
    """
    prediction: Optional[Tensor]              # (B, D) predicted next representation
    anomaly_score: Tensor                     # (B,) in [0, 1]
    tm_state: Optional[Any] = None
    sp_state: Optional[Any] = None
    promoted_patterns: int = 0


@dataclass
class ReasoningOutput:
    """Output from Dual-Process Reasoner.

    Invariants:
        y_sys1.shape[-1] == workspace_dim
        If used_sys2: y_sys2 is not None
        output = y_sys2 if used_sys2 else y_sys1
        conf_sys1 in [0, 1]
    """
    y_sys1: Tensor                            # (B, D) System 1 fast output
    conf_sys1: Tensor                         # (B, 1) System 1 confidence
    y_sys2: Optional[Tensor] = None           # (B, D) System 2 slow output
    used_sys2: bool = False
    output: Optional[Tensor] = None           # (B, D) final selected output
    trace: Optional[List[Tensor]] = None      # Reasoning step activations
    symbols: Optional[Dict[str, Tensor]] = None


@dataclass
class DecisionOutput:
    """Output from Active Inference decision system."""
    action_dist: Optional[Any] = None         # torch.distributions.Distribution
    action: Optional[Tensor] = None           # (B, A) selected action
    efe_terms: Optional[Dict[str, Tensor]] = None
    belief_state: Optional[Tensor] = None     # (B, S) updated belief
    action_logits: Optional[Tensor] = None    # (B, num_actions) for discrete


# ---------------------------------------------------------------------------
# System Output
# ---------------------------------------------------------------------------

@dataclass
class SystemDetails:
    """Detailed introspection — only populated when return_details=True.

    Fields may be None but keys are always present (stable schema).
    """
    encoder: Optional[Dict[str, Dict]] = None
    workspace: Optional[Dict] = None
    htm: Optional[Dict] = None
    reasoning: Optional[Dict] = None
    decision: Optional[Dict] = None
    meta: Optional[Dict] = None
    engram: Optional[Dict] = None


@dataclass
class SystemOutput:
    """Final output from BrainAI forward pass.

    Always-collected fields (cheap): output, confidence, modalities_used,
    reasoning_used, anomaly_score, inference_time_ms.

    Optional heavy traces (only when return_details=True): details.
    """
    # Always present
    output: Tensor                            # Task-dependent shape
    confidence: Tensor                        # (B, 1)
    modalities_used: List[str] = field(default_factory=list)
    reasoning_used: bool = False
    anomaly_score: Optional[Tensor] = None    # (B,) from HTM
    inference_time_ms: float = 0.0

    # Present only when return_details=True
    details: Optional[SystemDetails] = None


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------

@dataclass
class BrainAIState:
    """Consolidated state for all stateful modules.

    Serializable by torch.save. Each stage reads/writes its own slice.
    """
    wm_state: Optional[Any] = None            # Working memory (CfC/LTC/GRU)
    htm_state: Optional[Tuple] = None         # (tm_state, sp_state)
    snn_state: Optional[Dict[str, Tensor]] = None  # {layer_name: membrane_potential}
    belief_state: Optional[Tensor] = None     # (B, S) active inference belief
    eligibility_state: Optional[Tensor] = None
    rng_state: Optional[Dict[str, Any]] = None
    step_count: int = 0


# ---------------------------------------------------------------------------
# Pipeline Context (used internally by PipelinePlan)
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Mutable context passed through all pipeline stages."""
    batch: ModalityBatch = field(default_factory=ModalityBatch)
    config: Any = None  # BrainAIConfig (avoid circular import)

    # Accumulated results
    encoder_outputs: Dict[str, EncoderOutput] = field(default_factory=dict)
    repr: Optional[Tensor] = None              # Main representation (B,T,D) or (B,K,D)
    anomaly_score: Optional[Tensor] = None
    reasoning: Optional[ReasoningOutput] = None
    modulators: Optional[Dict[str, Tensor]] = None
    action: Optional[Tensor] = None
    decision: Optional[DecisionOutput] = None

    # State
    state: Optional[BrainAIState] = None
    new_state: Dict[str, Any] = field(default_factory=dict)

    # Details collection
    details: Dict[str, Any] = field(default_factory=dict)
    return_details: bool = False
    deterministic: bool = False

    # Telemetry
    telemetry: Optional[Any] = None           # TelemetrySink

    # Device / dtype
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    compute_dtype: torch.dtype = torch.float32

    # Timing
    _forward_start: float = 0.0

    def start_timing(self):
        self._forward_start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._forward_start) * 1000


# ---------------------------------------------------------------------------
# Contract Assertions
# ---------------------------------------------------------------------------

_CONTRACTS_ENABLED = True


def disable_contracts():
    """Disable runtime contract checks for production speed."""
    global _CONTRACTS_ENABLED
    _CONTRACTS_ENABLED = False


def assert_encoder_output(output: EncoderOutput, workspace_dim: int, device: torch.device):
    """Validate EncoderOutput contract."""
    if not _CONTRACTS_ENABLED:
        return
    assert output.feats.ndim == 3, f"feats must be 3D, got {output.feats.ndim}D"
    assert output.feats.shape[-1] == workspace_dim, (
        f"feats dim {output.feats.shape[-1]} != workspace_dim {workspace_dim}"
    )
    assert output.mask.shape[:2] == output.feats.shape[:2], "mask/feats shape mismatch"
    assert output.feats.device == device, f"feats on {output.feats.device}, expected {device}"


def assert_system_output(output: SystemOutput, batch_size: int, device: torch.device):
    """Validate SystemOutput contract."""
    if not _CONTRACTS_ENABLED:
        return
    assert output.output.shape[0] == batch_size, f"batch dim {output.output.shape[0]} != {batch_size}"
    assert output.confidence.shape == (batch_size, 1), f"confidence shape {output.confidence.shape}"
    assert output.output.device == device, f"output on {output.output.device}, expected {device}"
    assert not torch.isnan(output.output).any(), "NaN in output"
    assert not torch.isinf(output.output).any(), "Inf in output"
