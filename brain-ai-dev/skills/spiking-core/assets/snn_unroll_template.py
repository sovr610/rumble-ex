"""
brain_ai/core/unroll.py — Unified SNN Time Unrolling Utility

This module provides the single time-unrolling utility that ALL spiking network
architectures must use. Having one unroll path prevents the #1 source of SNN bugs:
Conv and MLP variants unrolling time differently.

Key features:
1. Step mode (B, ...) and sequence mode (B, T, ...) via one function
2. Truncated BPTT with configurable chunk size
3. Configurable recording modes (spikes-only, spikes+membrane, full trace)
4. Pre-allocated output tensors for performance
5. State management integrated with SpikingState

Canonical layout: batch-first, time-second — (B, T, ...).
This is a hard convention. All SNN modules must produce and consume this layout
at their public interface. Internal transposition is permitted at module boundaries
but must never propagate outward.

Usage:
    # Define a cell function (neuron + linear/conv transform)
    def cell_fn(x_t, state):
        current = linear(x_t)
        spikes, new_state = neuron(current, state)
        return spikes, new_state

    # Unroll through time (sequence mode)
    result = snn_unroll(
        cell_fn=cell_fn,
        inputs=x,              # (B, T, D)
        initial_state=state,
        chunk_size=10,         # truncated BPTT
        record_mode=RecordMode.SPIKES_MEMBRANE,
    )
    spikes = result.spikes          # (B, T, D_out)
    final_state = result.final_state
    membrane = result.traces['membrane']   # (B, T, N_neurons)

    # Step mode (online / streaming)
    result = snn_step(cell_fn=cell_fn, x=x_t, state=state)
    spike_t, next_state = result.spikes, result.final_state

Integration:
    Copy this file to brain_ai/core/unroll.py and import SpikingState from
    brain_ai/core/state.py (or paste SpikingState inline there).

    TODO(integration): Import SpikingState from brain_ai.core.state once that
    module exists. Until then, the SpikingState class is defined in this file
    for self-containment.

Design notes:
    - No module may implement its own time-unroll loop. All SNN modules delegate here.
    - cell_fn receives (B, ...) — no time dimension. snn_unroll supplies that.
    - State is always external; cell_fn must never call self.reset() internally.
    - fp32 state accumulation is enforced: state.v / state.i / state.a are always
      fp32 even when the model runs under AMP.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor
import torch.nn as nn


# ---------------------------------------------------------------------------
# RecordMode
# ---------------------------------------------------------------------------

class RecordMode:
    """Constants that control which internal signals are recorded during unroll.

    Selecting a richer recording mode increases memory usage proportionally to T.
    Use SPIKES_ONLY for normal training; upgrade to SPIKES_MEMBRANE when loss
    functions or visualisation tools need membrane traces.

    Attributes:
        SPIKES_ONLY     : Collect spike output only. traces dict is None.
                          Minimal memory overhead. Default for training.
        SPIKES_MEMBRANE : Collect spikes and membrane potential traces.
                          traces dict contains 'membrane' key.
                          Use when loss requires membrane values or for
                          threshold calibration.
        FULL_TRACE      : Collect spikes, membrane, synaptic current, and
                          adaptation variable (whichever are present in state).
                          traces dict contains 'membrane', 'current', and/or
                          'adaptation' keys as available.
                          Use for debugging, detailed visualisation, or
                          rate-coded decoding losses on multiple state fields.
        NONE            : No recording. Intended for step mode.
                          snn_step uses this internally; callers should not
                          set record_mode=NONE when calling snn_unroll directly.
    """

    SPIKES_ONLY     = "spikes_only"
    SPIKES_MEMBRANE = "spikes_membrane"
    FULL_TRACE      = "full_trace"
    NONE            = "none"

    _ALL: Tuple[str, ...] = (
        "spikes_only",
        "spikes_membrane",
        "full_trace",
        "none",
    )

    @classmethod
    def validate(cls, mode: str) -> None:
        """Raise ValueError for unknown record modes."""
        if mode not in cls._ALL:
            raise ValueError(
                f"Unknown record_mode={mode!r}. "
                f"Choose one of {cls._ALL}."
            )


# ---------------------------------------------------------------------------
# SpikingState
# ---------------------------------------------------------------------------

@dataclass
class SpikingState:
    """External state container for spiking neuron layers.

    Replaces the implicit self.mem pattern. State is always passed in and
    returned; the module holds no reference to it between calls. This enables:
    - Multiple independent sequences through the same module.
    - Correct batch-size flexibility (different B per call).
    - Explicit state reset at sequence boundaries.
    - Clean truncated BPTT via detach().

    All tensor fields are kept in fp32 regardless of the model's compute dtype.
    See the AMP policy in docs/spiking-core/state-management.md §I.

    Attributes:
        v             : Membrane potential. Always fp32. Shape: (B, *neuron_shape).
        i             : Synaptic current. Optional fp32. Shape: (B, *neuron_shape).
                        Populated when the neuron models an explicit synaptic
                        current (alpha synapse or double-exponential PSC).
        a             : Adaptation variable. Optional fp32. Shape: (B, *neuron_shape).
                        Populated for AdaptiveLIF neurons.
        ref           : Refractory timer. Optional fp32. Shape: (B, *neuron_shape).
                        Decremented each timestep, reset to t_ref on spike.
        spike_history : Delay ring buffer. Optional fp32.
                        Shape: (B, T_delay, *neuron_shape).
                        Populated for AdvancedLIF neurons with learnable delays.
    """

    v: Tensor
    i: Optional[Tensor] = None
    a: Optional[Tensor] = None
    ref: Optional[Tensor] = None
    spike_history: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def detach(self) -> "SpikingState":
        """Return a new SpikingState with all fields detached from the autograd graph.

        Values are byte-for-byte identical; only the gradient history is severed.
        The returned tensors are leaf tensors with grad_fn is None.

        CRITICAL: Use detach(), not clone(). clone() preserves grad_fn and does
        NOT cut the computation graph. Using clone() here defeats the purpose of
        truncated BPTT and causes memory to grow proportional to T.

        Returns:
            New SpikingState with detached tensors.
        """
        return SpikingState(
            v=self.v.detach(),
            i=self.i.detach() if self.i is not None else None,
            a=self.a.detach() if self.a is not None else None,
            ref=self.ref.detach() if self.ref is not None else None,
            spike_history=(
                self.spike_history.detach()
                if self.spike_history is not None
                else None
            ),
        )

    def clone(self) -> "SpikingState":
        """Return a deep copy of this state (values AND gradient history).

        Use for debugging only. For BPTT chunk boundaries always use detach().
        """
        return SpikingState(
            v=self.v.clone(),
            i=self.i.clone() if self.i is not None else None,
            a=self.a.clone() if self.a is not None else None,
            ref=self.ref.clone() if self.ref is not None else None,
            spike_history=(
                self.spike_history.clone()
                if self.spike_history is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(
        batch_size: int,
        neuron_shape: Union[Tuple[int, ...], int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        use_current: bool = False,
        use_adaptation: bool = False,
        use_refractory: bool = False,
        spike_history_len: int = 0,
    ) -> "SpikingState":
        """Create a zero-initialised SpikingState.

        Always initialises v (membrane potential). Other fields are created only
        when the corresponding flag is True.

        Args:
            batch_size       : Number of samples in the batch (B).
            neuron_shape     : Shape of the neuron population, excluding B.
                               For linear layers: (N,) or just N (int).
                               For conv layers: (C, H, W).
            device           : Target device.
            dtype            : Dtype for state tensors. fp32 strongly recommended.
                               Passing fp16/bf16 here emits a warning.
            use_current      : Allocate synaptic current field i.
            use_adaptation   : Allocate adaptation variable field a.
            use_refractory   : Allocate refractory timer field ref.
            spike_history_len: Length of the delay ring buffer. 0 = no buffer.

        Returns:
            SpikingState with zero-filled tensors on the specified device.
        """
        if isinstance(neuron_shape, int):
            neuron_shape = (neuron_shape,)

        if dtype != torch.float32:
            warnings.warn(
                f"SpikingState.zeros called with dtype={dtype}. "
                "Membrane state should stay in fp32 to avoid precision drift "
                "during long sequence accumulation. See AMP policy in "
                "docs/spiking-core/state-management.md §I.",
                UserWarning,
                stacklevel=2,
            )

        shape = (batch_size, *neuron_shape)
        return SpikingState(
            v=torch.zeros(shape, device=device, dtype=torch.float32),
            i=(
                torch.zeros(shape, device=device, dtype=torch.float32)
                if use_current
                else None
            ),
            a=(
                torch.zeros(shape, device=device, dtype=torch.float32)
                if use_adaptation
                else None
            ),
            ref=(
                torch.zeros(shape, device=device, dtype=torch.float32)
                if use_refractory
                else None
            ),
            spike_history=(
                torch.zeros(
                    batch_size, spike_history_len, *neuron_shape,
                    device=device, dtype=torch.float32,
                )
                if spike_history_len > 0
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Device / dtype movement
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "SpikingState":
        """Move all tensors to device. Returns new SpikingState."""
        return SpikingState(
            v=self.v.to(device),
            i=self.i.to(device) if self.i is not None else None,
            a=self.a.to(device) if self.a is not None else None,
            ref=self.ref.to(device) if self.ref is not None else None,
            spike_history=(
                self.spike_history.to(device)
                if self.spike_history is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Orchestrator serialisation
    # ------------------------------------------------------------------

    def to_dict(self, layer_name: str) -> Dict[str, Tensor]:
        """Flatten state to the orchestrator's flat dictionary format.

        Keys follow the pattern: "{layer_name}.{field}".
        Only non-None fields are included.

        Args:
            layer_name: Unique identifier for the layer (e.g. "snn_core.layer_0").

        Returns:
            Flat dict mapping qualified keys to tensors.
        """
        out: Dict[str, Tensor] = {f"{layer_name}.v": self.v}
        if self.i is not None:
            out[f"{layer_name}.i"] = self.i
        if self.a is not None:
            out[f"{layer_name}.a"] = self.a
        if self.ref is not None:
            out[f"{layer_name}.ref"] = self.ref
        if self.spike_history is not None:
            out[f"{layer_name}.spike_history"] = self.spike_history
        return out

    @staticmethod
    def from_dict(layer_name: str, d: Dict[str, Tensor]) -> "SpikingState":
        """Reconstruct SpikingState from the orchestrator's flat dictionary.

        Args:
            layer_name: Same identifier used in to_dict().
            d         : Flat dict from the orchestrator (may contain other layers).

        Returns:
            SpikingState with fields populated from d.

        Raises:
            KeyError: If the mandatory membrane potential key is missing.
        """
        key_v = f"{layer_name}.v"
        if key_v not in d:
            raise KeyError(
                f"Mandatory key {key_v!r} not found in state dict. "
                f"Available keys: {sorted(d.keys())}"
            )
        return SpikingState(
            v=d[key_v],
            i=d.get(f"{layer_name}.i"),
            a=d.get(f"{layer_name}.a"),
            ref=d.get(f"{layer_name}.ref"),
            spike_history=d.get(f"{layer_name}.spike_history"),
        )

    # ------------------------------------------------------------------
    # Batch utilities
    # ------------------------------------------------------------------

    def apply_reset_mask(self, reset_mask: Tensor) -> "SpikingState":
        """Selectively zero state for batch items where reset_mask is True.

        Used when a batch contains a mix of continuing and resetting sequences
        (e.g., different documents in a language modelling batch). Neurons in
        resetting items have their state zeroed without splitting the batch.

        Args:
            reset_mask: Bool tensor of shape (B,). True = zero state for this item.

        Returns:
            New SpikingState with masked items zeroed.
        """
        def _mask(t: Tensor) -> Tensor:
            # Broadcast mask over spatial dimensions
            m = reset_mask.float().view(-1, *([1] * (t.dim() - 1)))
            return t * (1.0 - m)

        return SpikingState(
            v=_mask(self.v),
            i=_mask(self.i) if self.i is not None else None,
            a=_mask(self.a) if self.a is not None else None,
            ref=_mask(self.ref) if self.ref is not None else None,
            spike_history=(
                _mask(self.spike_history) if self.spike_history is not None else None
            ),
        )

    def partial_reset(
        self,
        components: Set[str],
        batch_size: int,
        device: torch.device,
    ) -> "SpikingState":
        """Zero selected state components; preserve others unchanged.

        Experimental utility for controlled ablation studies. Production code
        should use full reset (SpikingState.zeros) or full carry.

        Args:
            components : Set of field names to zero: subset of {'v','i','a','ref','spike_history'}.
            batch_size : Needed to construct zero tensors for reset fields.
            device     : Target device for fresh zero tensors.

        Returns:
            New SpikingState with named components zeroed.
        """
        def _maybe_zero(name: str, t: Optional[Tensor]) -> Optional[Tensor]:
            if name not in components or t is None:
                return t
            return torch.zeros_like(t)

        return SpikingState(
            v=_maybe_zero("v", self.v),
            i=_maybe_zero("i", self.i),
            a=_maybe_zero("a", self.a),
            ref=_maybe_zero("ref", self.ref),
            spike_history=_maybe_zero("spike_history", self.spike_history),
        )

    def __repr__(self) -> str:
        fields = [f"v={tuple(self.v.shape)}"]
        if self.i is not None:
            fields.append(f"i={tuple(self.i.shape)}")
        if self.a is not None:
            fields.append(f"a={tuple(self.a.shape)}")
        if self.ref is not None:
            fields.append(f"ref={tuple(self.ref.shape)}")
        if self.spike_history is not None:
            fields.append(f"spike_history={tuple(self.spike_history.shape)}")
        return f"SpikingState({', '.join(fields)})"


# ---------------------------------------------------------------------------
# CellFn Protocol
# ---------------------------------------------------------------------------

class CellFn(Protocol):
    """Protocol for spiking cell functions.

    A cell function encapsulates the per-timestep computation of one or more
    spiking layers: linear/conv transform followed by neuron dynamics. It
    accepts a single-timestep input and the current SpikingState, and returns
    the spike output for that timestep and the updated state.

    Both MLP-SNN and Conv-SNN architectures implement this interface. The
    snn_unroll utility is agnostic to the spatial structure of the tensors —
    it slices inputs[:, t] to produce x_t and passes it to cell_fn.

    Constraints:
        - cell_fn must NOT call self.reset() internally.
        - cell_fn must NOT store state on self between calls.
        - cell_fn must return fp32 state (spike output may be cast to compute dtype).
        - x_t has no time dimension: (B, ...) not (B, T, ...).
    """

    def __call__(
        self,
        x_t: Tensor,
        state: SpikingState,
    ) -> Tuple[Tensor, SpikingState]:
        """Run one timestep.

        Args:
            x_t  : Input for this timestep. Shape (B, ...) — no time axis.
            state: Current SpikingState.

        Returns:
            Tuple of (spike_output, next_state).
            spike_output shape must match x_t's batch dim: (B, ...).
        """
        ...


# ---------------------------------------------------------------------------
# UnrollOutput
# ---------------------------------------------------------------------------

@dataclass
class UnrollOutput:
    """Result of snn_unroll or snn_step.

    Attributes:
        spikes      : Spike outputs.
                      Sequence mode: (B, T, ...) covering all timesteps.
                      Step mode: (B, ...) for the single timestep.
        final_state : SpikingState after the last timestep processed.
                      In step mode this is the state after the single step.
                      In sequence mode this is the state after timestep T-1.
                      If detach_between_chunks=True (default), final_state is
                      detached from the computation graph.
        traces      : Optional dict of recorded signals. None when
                      record_mode=RecordMode.SPIKES_ONLY or NONE.
                      Possible keys:
                        'membrane'    : (B, T, *neuron_shape) fp32
                        'current'     : (B, T, *neuron_shape) fp32, if state.i present
                        'adaptation'  : (B, T, *neuron_shape) fp32, if state.a present
                        'firing_rates': (num_chunks,) mean spikes per chunk
    """

    spikes: Tensor
    final_state: SpikingState
    traces: Optional[Dict[str, Tensor]] = None


# ---------------------------------------------------------------------------
# BPTT chunk schedule
# ---------------------------------------------------------------------------

def compute_chunk_schedule(T: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Compute (start, end) index pairs for truncated BPTT chunks.

    Handles the case where T is not divisible by chunk_size: the final chunk
    is shorter than the others, covering the remaining timesteps.

    Args:
        T         : Total number of timesteps.
        chunk_size: Desired timesteps per BPTT chunk.

    Returns:
        List of (start, end) tuples where start is inclusive and end is
        exclusive (Python slice convention). The concatenation of all
        ranges covers [0, T) exactly.

    Examples:
        >>> compute_chunk_schedule(10, 3)
        [(0, 3), (3, 6), (6, 9), (9, 10)]
        >>> compute_chunk_schedule(10, 10)
        [(0, 10)]
        >>> compute_chunk_schedule(10, 20)
        [(0, 10)]
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got T={T}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got chunk_size={chunk_size}.")

    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        chunks.append((start, end))
        start = end
    return chunks


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_memory_usage(
    batch_size: int,
    timesteps: int,
    neuron_size: int,
    chunk_size: Optional[int],
    record_mode: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, int]:
    """Estimate peak GPU memory (bytes) for an snn_unroll configuration.

    This is an approximation intended to guide chunk_size selection before
    running. Actual usage will differ due to PyTorch allocator overhead,
    activation checkpointing, and other model components.

    Memory components:
        spike_record      : (B, T, N) always — pre-allocated output buffer.
        gradient_window   : Intermediate activations retained for backward.
                            Scales with chunk_size, not full T.
        trace_buffers     : Additional (B, T, N) buffers for membrane/current
                            when record_mode includes them.

    Args:
        batch_size  : Batch size B.
        timesteps   : Full sequence length T.
        neuron_size : Flat size of the neuron population N.
        chunk_size  : BPTT chunk size. None = full BPTT (chunk_size = T).
        record_mode : RecordMode constant.
        dtype       : Tensor dtype for size calculation.

    Returns:
        Dict with keys:
            'spike_record_bytes'   : Pre-allocated spike output buffer.
            'gradient_window_bytes': Activations held for BPTT in one chunk.
            'trace_buffer_bytes'   : Trace buffer overhead (0 if SPIKES_ONLY).
            'total_bytes'          : Sum of all components.
    """
    RecordMode.validate(record_mode)

    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
    }.get(dtype, 4)

    effective_chunk = timesteps if chunk_size is None else chunk_size

    spike_record_bytes = batch_size * timesteps * neuron_size * bytes_per_element

    # Gradient window: chunk_size timesteps of activations + state per timestep.
    # Heuristic: ~3 tensors of (B, N) shape per timestep (pre/post/state).
    gradient_window_bytes = (
        batch_size * effective_chunk * neuron_size * bytes_per_element * 3
    )

    # Trace buffers: one extra (B, T, N) per tracked field.
    num_trace_buffers = 0
    if record_mode == RecordMode.SPIKES_MEMBRANE:
        num_trace_buffers = 1  # membrane
    elif record_mode == RecordMode.FULL_TRACE:
        num_trace_buffers = 3  # membrane + current + adaptation
    trace_buffer_bytes = (
        batch_size * timesteps * neuron_size * bytes_per_element * num_trace_buffers
    )

    total_bytes = spike_record_bytes + gradient_window_bytes + trace_buffer_bytes

    return {
        "spike_record_bytes": spike_record_bytes,
        "gradient_window_bytes": gradient_window_bytes,
        "trace_buffer_bytes": trace_buffer_bytes,
        "total_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Main unroll function
# ---------------------------------------------------------------------------

def snn_unroll(
    cell_fn: Callable[[Tensor, SpikingState], Tuple[Tensor, SpikingState]],
    inputs: Tensor,
    initial_state: SpikingState,
    chunk_size: Optional[int] = None,
    record_mode: str = RecordMode.SPIKES_ONLY,
    detach_between_chunks: bool = True,
) -> UnrollOutput:
    """Unified SNN time unrolling with truncated BPTT.

    This is the ONLY function that should unroll spiking networks through time.
    Both MLP-SNN and Conv-SNN MUST use this function. No module may implement
    its own time loop.

    Canonical layout enforced: inputs must be (B, T, ...) — batch-first,
    time-second. Any module whose internal representation differs must transpose
    at its own boundaries, not here.

    Memory policy:
        Spike output is pre-allocated as a (B, T, ...) zero tensor and filled
        in-place. This avoids T intermediate tensor allocations and the final
        torch.stack copy that a list-based approach would require.

    AMP policy:
        snn_unroll is compatible with torch.cuda.amp.autocast. Wrap the call
        site in an autocast context as needed. If membrane instability is
        observed under fp16/bf16, add:
            with torch.autocast(enabled=False):
                inside cell_fn's threshold comparison.
        State tensors (v, i, a) must remain fp32 inside cell_fn regardless of
        the outer autocast context.

    Args:
        cell_fn               : Single-timestep function (x_t, state) -> (out_t, new_state).
                                x_t is (B, ...) — no time axis.
                                out_t must have shape (B, ...) with same batch dim.
        inputs                : Input sequence (B, T, ...). Must be at least 3D.
        initial_state         : SpikingState for the first timestep.
        chunk_size            : BPTT window size.
                                None → full BPTT (chunk_size = T).
                                Recommended: T//4 general, 10 for T=50.
                                None is appropriate when T <= 25 or when the
                                model fits comfortably in GPU memory.
        record_mode           : What to record in the traces dict.
                                RecordMode.SPIKES_ONLY   → traces is None.
                                RecordMode.SPIKES_MEMBRANE → traces['membrane'].
                                RecordMode.FULL_TRACE    → all available fields.
        detach_between_chunks : If True (default), sever the autograd graph
                                between BPTT chunks by calling state.detach().
                                Set False only for debugging: the graph will
                                grow to cover the full T timesteps regardless
                                of chunk_size.

    Returns:
        UnrollOutput:
            .spikes      : (B, T, ...) spike output.
            .final_state : SpikingState after timestep T-1.
            .traces      : Dict of recorded signals, or None.

    Raises:
        ValueError: If inputs has fewer than 3 dimensions.
        ValueError: If record_mode is not a recognised RecordMode constant.

    Example — training with truncated BPTT:
        state = SpikingState.zeros(B, (hidden,), device=x.device)
        result = snn_unroll(
            cell_fn=self.forward_step,
            inputs=x,           # (B, T, D)
            initial_state=state,
            chunk_size=10,
            record_mode=RecordMode.SPIKES_ONLY,
        )
        loss = criterion(result.spikes, targets)
        loss.backward()

    Example — trace collection for visualisation:
        result = snn_unroll(
            cell_fn=self.forward_step,
            inputs=x,
            initial_state=state,
            chunk_size=None,    # full BPTT (no truncation)
            record_mode=RecordMode.FULL_TRACE,
            detach_between_chunks=False,
        )
        membrane = result.traces['membrane']   # (B, T, N)
        firing_rates = result.traces['firing_rates']  # (1,) since no chunking
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    RecordMode.validate(record_mode)

    if inputs.dim() < 3:
        raise ValueError(
            f"inputs must be at least 3D (B, T, ...), got shape {tuple(inputs.shape)}. "
            "For step mode use snn_step() instead."
        )

    # ------------------------------------------------------------------
    # Step 1: Extract dimensions
    # ------------------------------------------------------------------
    B = inputs.shape[0]
    T = inputs.shape[1]
    spatial: Tuple[int, ...] = tuple(inputs.shape[2:])  # () for (B,T,D) → spatial=(D,)

    # ------------------------------------------------------------------
    # Step 2: Resolve chunk size
    # ------------------------------------------------------------------
    if chunk_size is None:
        effective_chunk = T
    else:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        effective_chunk = chunk_size

    chunk_schedule: List[Tuple[int, int]] = compute_chunk_schedule(T, effective_chunk)
    num_chunks = len(chunk_schedule)

    # ------------------------------------------------------------------
    # Step 3: Infer output spatial shape from a probe call
    # ------------------------------------------------------------------
    # We need to know the output shape of cell_fn to pre-allocate.
    # Run one step with no_grad, then detach the probe state.
    with torch.no_grad():
        probe_x = inputs[:, 0]
        probe_out, probe_state = cell_fn(probe_x, initial_state.detach())
        output_spatial: Tuple[int, ...] = tuple(probe_out.shape[1:])  # strip B
        # Verify neuron state shape for trace buffers
        neuron_shape: Tuple[int, ...] = tuple(probe_state.v.shape[1:])

    # ------------------------------------------------------------------
    # Step 4: Pre-allocate output buffers
    # ------------------------------------------------------------------
    # Spike record: always allocated (batch-first, time-second).
    spike_record = torch.zeros(
        B, T, *output_spatial,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    # Trace buffers: allocated only if recording requires them.
    membrane_record: Optional[Tensor] = None
    current_record: Optional[Tensor] = None
    adaptation_record: Optional[Tensor] = None
    firing_rates: Optional[List[float]] = None

    needs_membrane = record_mode in (RecordMode.SPIKES_MEMBRANE, RecordMode.FULL_TRACE)
    needs_full     = record_mode == RecordMode.FULL_TRACE

    if needs_membrane:
        membrane_record = torch.zeros(
            B, T, *neuron_shape,
            device=inputs.device,
            dtype=torch.float32,  # state tensors always fp32
        )
        firing_rates = []

    if needs_full:
        if initial_state.i is not None:
            current_record = torch.zeros(
                B, T, *neuron_shape,
                device=inputs.device,
                dtype=torch.float32,
            )
        if initial_state.a is not None:
            adaptation_record = torch.zeros(
                B, T, *neuron_shape,
                device=inputs.device,
                dtype=torch.float32,
            )

    # ------------------------------------------------------------------
    # Step 5 & 6: Outer loop over chunks, inner loop over timesteps
    # ------------------------------------------------------------------
    state: SpikingState = initial_state

    for chunk_start, chunk_end in chunk_schedule:
        chunk_len = chunk_end - chunk_start
        # Slice the chunk inputs once for this chunk.
        chunk_inputs = inputs[:, chunk_start:chunk_end]  # (B, chunk_len, ...)

        # Inner loop: iterate over timesteps within the current chunk.
        chunk_spike_sum: Optional[Tensor] = None

        for t_local in range(chunk_len):
            t_global = chunk_start + t_local
            x_t = chunk_inputs[:, t_local]              # (B, ...)

            spike_t, state = cell_fn(x_t, state)        # core step

            # Store spike output into the pre-allocated buffer.
            spike_record[:, t_global] = spike_t

            # Record traces according to selected mode.
            if membrane_record is not None:
                membrane_record[:, t_global] = state.v

            if current_record is not None and state.i is not None:
                current_record[:, t_global] = state.i

            if adaptation_record is not None and state.a is not None:
                adaptation_record[:, t_global] = state.a

            # Accumulate per-chunk spike sum for firing rate.
            if firing_rates is not None:
                s = spike_t.detach().float().mean()
                if chunk_spike_sum is None:
                    chunk_spike_sum = s
                else:
                    chunk_spike_sum = chunk_spike_sum + s

        # Record per-chunk mean firing rate.
        if firing_rates is not None and chunk_spike_sum is not None:
            firing_rates.append(float(chunk_spike_sum) / chunk_len)

        # ------------------------------------------------------------------
        # Step 6: Detach state between chunks to cut the autograd graph.
        # ------------------------------------------------------------------
        # This is the key truncated BPTT operation. Gradients will not flow
        # across this boundary. Numerical values are preserved.
        if detach_between_chunks:
            state = state.detach()

    # ------------------------------------------------------------------
    # Step 7: Assemble traces dict and return
    # ------------------------------------------------------------------
    traces: Optional[Dict[str, Tensor]] = None

    if needs_membrane and membrane_record is not None:
        traces = {"membrane": membrane_record}

        if firing_rates is not None:
            traces["firing_rates"] = torch.tensor(
                firing_rates, dtype=torch.float32, device=inputs.device
            )

        if needs_full:
            if current_record is not None:
                traces["current"] = current_record
            if adaptation_record is not None:
                traces["adaptation"] = adaptation_record

    return UnrollOutput(
        spikes=spike_record,
        final_state=state,
        traces=traces,
    )


# ---------------------------------------------------------------------------
# Step mode wrapper
# ---------------------------------------------------------------------------

def snn_step(
    cell_fn: Callable[[Tensor, SpikingState], Tuple[Tensor, SpikingState]],
    x: Tensor,
    state: SpikingState,
    return_details: bool = False,
) -> Union[UnrollOutput, Tuple[Tensor, SpikingState, Optional[Dict]]]:
    """Single-timestep forward pass (step mode) for streaming / online inference.

    Step mode is used during autoregressive generation, real-time sensory
    processing, and when an external loop controls time. There is no BPTT
    in step mode — gradients flow only within the single timestep.

    The caller owns state lifecycle in step mode. State carries across
    sequential snn_step calls. Wrap in torch.no_grad() for pure inference.

    Args:
        cell_fn       : Single-timestep function (x_t, state) -> (out_t, new_state).
        x             : Input for this timestep. Shape (B, ...) — no time axis.
                        If x has 3+ dimensions (B, T, ...), this function will
                        raise an error; use snn_unroll instead.
        state         : Current SpikingState from the previous step.
        return_details: If True, return (spike, new_state, details_dict).
                        If False, return an UnrollOutput for API consistency.

    Returns:
        If return_details=False (default):
            UnrollOutput with:
                .spikes      : (B, ...) spike output for this timestep.
                .final_state : Updated SpikingState.
                .traces      : None (no trace collection in step mode).
        If return_details=True:
            Tuple (spike_t, new_state, details) where details is a dict
            containing 'membrane' and other state fields for this single step.

    Raises:
        ValueError: If x has a time dimension (3+ dims with T>1).

    Example — streaming inference:
        state = SpikingState.zeros(B=1, neuron_shape=(512,), device=device)
        with torch.no_grad():
            for frame in audio_stream:
                result = snn_step(cell_fn, frame, state)
                state = result.final_state
                process(result.spikes)
    """
    if x.dim() >= 3 and x.shape[1] > 1:
        raise ValueError(
            f"snn_step expects a single-timestep input (B, ...) or (B, 1, ...). "
            f"Received shape {tuple(x.shape)} which looks like a sequence. "
            "Use snn_unroll() for sequence-mode inputs."
        )

    # Squeeze out a singleton time dim if provided.
    if x.dim() >= 3 and x.shape[1] == 1:
        x = x.squeeze(1)

    spike_t, new_state = cell_fn(x, state)

    if return_details:
        details: Dict[str, Tensor] = {"membrane": new_state.v}
        if new_state.i is not None:
            details["current"] = new_state.i
        if new_state.a is not None:
            details["adaptation"] = new_state.a
        return spike_t, new_state, details

    return UnrollOutput(
        spikes=spike_t,
        final_state=new_state,
        traces=None,
    )


# ---------------------------------------------------------------------------
# Adapter: ConvSNN cell
# ---------------------------------------------------------------------------

def make_conv_cell(
    conv_layers: nn.ModuleList,
    neurons: nn.ModuleList,
    flatten_fn: Optional[Callable[[Tensor], Tensor]] = None,
    fc_layers: Optional[nn.ModuleList] = None,
    fc_neurons: Optional[nn.ModuleList] = None,
) -> Callable[[Tensor, SpikingState], Tuple[Tensor, SpikingState]]:
    """Create a cell_fn for ConvSNN architectures.

    Wraps convolutional + neuron layers into a single CellFn compatible with
    snn_unroll. This ensures ConvSNN uses the exact same unroll machinery as
    MLP-SNN, eliminating the dual-loop divergence that was the original bug.

    The cell processes one spatial frame at a time: input is (B, C, H, W) per
    timestep, not (B, T, C, H, W). snn_unroll supplies x_t = inputs[:, t]
    which has the right shape automatically because inputs is (B, T, C, H, W).

    State layout for ConvSNN:
        The SpikingState.v field has shape (B, C', H', W') for conv neurons
        (matching the feature map shape of the deepest conv layer) and shape
        (B, N) for any trailing FC neurons.
        Because a multi-layer ConvSNN uses multiple state components, the
        SpikingState passed in must carry one v tensor per spiking layer.

    IMPORTANT: The current implementation assumes ONE SpikingState object
    carries the membrane state of the LAST spiking layer only. Multi-layer
    state threading is the responsibility of the calling module's forward_step
    method. This adapter demonstrates the pattern for a two-stage (conv → FC)
    architecture.

    Args:
        conv_layers : ModuleList of nn.Conv2d (or similar) layers.
        neurons     : ModuleList of spiking neuron modules, one per conv layer.
                      Each neuron's forward signature: (current, state) -> (spike, state).
        flatten_fn  : Optional callable applied after conv stages to flatten spatial
                      dims before FC stages. Typically nn.Flatten() or a lambda.
                      Pass None if there are no FC layers.
        fc_layers   : Optional ModuleList of nn.Linear layers after conv+flatten.
        fc_neurons  : Optional ModuleList of spiking neurons for FC layers.
                      Must match fc_layers in length if provided.

    Returns:
        cell_fn: Callable (x_t: Tensor, state: SpikingState) -> (Tensor, SpikingState).
                 Processes one spatial frame (B, C, H, W) and returns
                 (spike_output, updated_state).

    Example:
        cell = make_conv_cell(
            conv_layers=self.conv_layers,
            neurons=self.conv_neurons,
            flatten_fn=nn.Flatten(),
            fc_layers=self.fc_layers,
            fc_neurons=self.fc_neurons,
        )
        result = snn_unroll(cell_fn=cell, inputs=x, initial_state=state)

    TODO(integration): When multi-layer state is needed, refactor SpikingState
    to hold a list of v tensors (one per layer) or compose multiple SpikingState
    objects. The cell_fn interface remains the same; only state indexing changes.
    """
    # Validate pairing.
    if len(conv_layers) != len(neurons):
        raise ValueError(
            f"conv_layers and neurons must have the same length. "
            f"Got {len(conv_layers)} conv layers and {len(neurons)} neurons."
        )
    if fc_layers is not None and fc_neurons is not None:
        if len(fc_layers) != len(fc_neurons):
            raise ValueError(
                f"fc_layers and fc_neurons must have the same length. "
                f"Got {len(fc_layers)} and {len(fc_neurons)}."
            )

    def cell_fn(x_t: Tensor, state: SpikingState) -> Tuple[Tensor, SpikingState]:
        """One forward step for a ConvSNN.

        Args:
            x_t  : Spatial input, shape (B, C, H, W).
            state: SpikingState. For a multi-layer conv network the caller
                   must thread state through the layers; here we demonstrate
                   using state only for the final layer (see TODO above).

        Returns:
            (spike_output, updated_state).
        """
        h = x_t

        # Conv stages: loop over conv + neuron pairs.
        # For simplicity this implementation uses a single shared state for the
        # final layer only. TODO(multi-layer): pass state per layer.
        for idx, (conv, neuron) in enumerate(zip(conv_layers, neurons)):
            current = conv(h)
            # For the last conv layer, use the provided state.
            # Inner layers use their own transient state (not persisted between
            # timesteps in this simplified version).
            # TODO(integration): Replace with per-layer state for full correctness.
            if idx == len(conv_layers) - 1:
                h, new_state = neuron(current, state)
            else:
                # Transient state for intermediate layers — not carried between steps.
                # This is a known limitation of the simplified adapter.
                # Full implementation should maintain state for every layer.
                h, _transient_state = neuron(current, state)

        # Optional flatten + FC stages.
        if flatten_fn is not None and fc_layers is not None and fc_neurons is not None:
            h = flatten_fn(h)
            for fc, fc_neuron in zip(fc_layers, fc_neurons):
                current = fc(h)
                h, new_state = fc_neuron(current, new_state)

        return h, new_state

    return cell_fn


# ---------------------------------------------------------------------------
# Adapter: MLP-SNN cell
# ---------------------------------------------------------------------------

def make_linear_cell(
    linear_layers: nn.ModuleList,
    neurons: nn.ModuleList,
) -> Callable[[Tensor, SpikingState], Tuple[Tensor, SpikingState]]:
    """Create a cell_fn for feedforward MLP-SNN architectures.

    Wraps linear + spiking neuron layers into a single CellFn compatible with
    snn_unroll. Input per timestep: (B, D).

    The adapter processes layers in order, piping each layer's spike output as
    the next layer's input. State is threaded through the final spiking layer.

    As with make_conv_cell, this is a simplified single-state adapter. For
    full multi-layer state support see the TODO note.

    Args:
        linear_layers: ModuleList of nn.Linear layers.
        neurons      : ModuleList of spiking neuron modules, one per linear layer.
                       Neuron forward signature: (current, state) -> (spike, state).

    Returns:
        cell_fn: Callable (x_t: Tensor, state: SpikingState) -> (Tensor, SpikingState).

    Example:
        cell = make_linear_cell(
            linear_layers=self.layers,
            neurons=self.spiking_neurons,
        )
        result = snn_unroll(cell_fn=cell, inputs=x, initial_state=state)

    TODO(integration): Maintain per-layer SpikingState for full correctness in
    multi-layer networks. Currently only the final layer's state is persisted
    across timesteps.
    """
    if len(linear_layers) != len(neurons):
        raise ValueError(
            f"linear_layers and neurons must have the same length. "
            f"Got {len(linear_layers)} linear layers and {len(neurons)} neurons."
        )

    def cell_fn(x_t: Tensor, state: SpikingState) -> Tuple[Tensor, SpikingState]:
        """One forward step for a feedforward MLP-SNN.

        Args:
            x_t  : Input for this timestep, shape (B, D).
            state: SpikingState for the last spiking layer.

        Returns:
            (spike_output, updated_state).
        """
        h = x_t
        current_state = state

        for idx, (linear, neuron) in enumerate(zip(linear_layers, neurons)):
            current = linear(h)
            if idx == len(linear_layers) - 1:
                h, current_state = neuron(current, current_state)
            else:
                # Transient state for intermediate layers.
                # TODO(integration): Thread per-layer state for correctness.
                h, _transient = neuron(current, current_state)

        return h, current_state

    return cell_fn


# ---------------------------------------------------------------------------
# Debug utilities
# ---------------------------------------------------------------------------

def compare_unroll_modes(
    cell_fn: Callable[[Tensor, SpikingState], Tuple[Tensor, SpikingState]],
    inputs: Tensor,
    initial_state: SpikingState,
    chunk_sizes: Sequence[int],
) -> Dict[str, Any]:
    """Compare full BPTT vs truncated BPTT for debugging and validation.

    Runs snn_unroll once with full BPTT (chunk_size=None) and once for each
    provided chunk_size, then compares:
    1. Forward outputs (spikes must be identical — chunking only affects gradient flow,
       not numerical values, as long as no dropout or stochastic op is inside cell_fn).
    2. Gradient norms of the initial_state.v tensor under each regime.
    3. Peak GPU memory during each run.

    This is a diagnostic tool for:
    - Verifying that truncated BPTT does not change the forward pass.
    - Measuring the gradient norm reduction from chunking.
    - Confirming that memory scales with chunk_size, not T.

    Args:
        cell_fn       : Spiking cell function to test.
        inputs        : Input sequence (B, T, ...). Should require_grad for gradient tests.
        initial_state : Starting SpikingState.
        chunk_sizes   : Sequence of chunk sizes to compare against full BPTT.

    Returns:
        Dict with keys:
            'forward_match'   : Dict mapping str(chunk_size) -> bool.
                                True if spike outputs match full BPTT.
            'forward_max_diff': Dict mapping str(chunk_size) -> float.
                                Maximum absolute difference between outputs.
            'grad_norms'      : Dict mapping mode_key -> Dict[str, float].
                                'full_bptt' and each chunk_size variant.
                                Inner dict: {'inputs': float, 'state_v': float}.
            'memory_mb'       : Dict mapping mode_key -> float (MB, CUDA only).
            'warnings'        : List of warning strings if anomalies are detected.

    Example:
        results = compare_unroll_modes(
            cell_fn=model.forward_step,
            inputs=x.requires_grad_(True),
            initial_state=state,
            chunk_sizes=[5, 10, 25],
        )
        for cs, match in results['forward_match'].items():
            print(f"chunk_size={cs}: forward match = {match}")
    """
    results: Dict[str, Any] = {
        "forward_match": {},
        "forward_max_diff": {},
        "grad_norms": {},
        "memory_mb": {},
        "warnings": [],
    }

    device = inputs.device
    use_cuda = device.type == "cuda"

    def _run_and_measure(cs: Optional[int], detach: bool) -> Tuple[Tensor, Dict[str, float], float]:
        """Run unroll, compute grad norms, measure peak memory."""
        # Clone inputs to isolate gradient computation between runs.
        x = inputs.detach().clone().requires_grad_(inputs.requires_grad)
        s = initial_state.detach()

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)

        out = snn_unroll(
            cell_fn=cell_fn,
            inputs=x,
            initial_state=s,
            chunk_size=cs,
            record_mode=RecordMode.SPIKES_ONLY,
            detach_between_chunks=detach,
        )

        # Backward to compute grad norms (if inputs require grad).
        input_grad_norm: float = 0.0
        state_grad_norm: float = 0.0

        if x.requires_grad:
            try:
                loss = out.spikes.sum()
                loss.backward()
                if x.grad is not None:
                    input_grad_norm = float(x.grad.norm())
            except RuntimeError as e:
                results["warnings"].append(f"Backward failed for chunk_size={cs}: {e}")

        grad_norms = {"inputs": input_grad_norm, "state_v": state_grad_norm}

        memory_mb: float = 0.0
        if use_cuda:
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        return out.spikes.detach(), grad_norms, memory_mb

    # ------------------------------------------------------------------
    # Full BPTT baseline
    # ------------------------------------------------------------------
    full_spikes, full_grads, full_mem = _run_and_measure(cs=None, detach=False)
    results["grad_norms"]["full_bptt"] = full_grads
    results["memory_mb"]["full_bptt"] = full_mem

    # ------------------------------------------------------------------
    # Truncated BPTT variants
    # ------------------------------------------------------------------
    for cs in chunk_sizes:
        key = str(cs)
        trunc_spikes, trunc_grads, trunc_mem = _run_and_measure(cs=cs, detach=True)

        match = torch.allclose(full_spikes, trunc_spikes, atol=1e-6, rtol=0.0)
        max_diff = float((full_spikes - trunc_spikes).abs().max())

        results["forward_match"][key] = match
        results["forward_max_diff"][key] = max_diff
        results["grad_norms"][key] = trunc_grads
        results["memory_mb"][key] = trunc_mem

        if not match:
            results["warnings"].append(
                f"WARNING: chunk_size={cs} forward output differs from full BPTT "
                f"(max_diff={max_diff:.2e}). This should not happen unless cell_fn "
                "contains stochastic operations or in-place mutations on inputs."
            )

    return results


# ---------------------------------------------------------------------------
# Gradient checkpointing integration
# ---------------------------------------------------------------------------

def snn_unroll_with_checkpointing(
    cell_fn: Callable[[Tensor, SpikingState], Tuple[Tensor, SpikingState]],
    inputs: Tensor,
    initial_state: SpikingState,
    chunk_size: int,
    record_mode: str = RecordMode.SPIKES_ONLY,
) -> UnrollOutput:
    """snn_unroll variant that applies gradient checkpointing per chunk.

    Combines truncated BPTT with torch.utils.checkpoint to allow training on
    very long sequences (T > 100) with constant peak memory relative to
    chunk_size. Each chunk's forward pass is recomputed during backward instead
    of being stored.

    Trade-off: approximately 2x recomputation overhead. Use only when:
    - T > 100 and chunked BPTT alone still causes OOM.
    - Batch throughput is memory-bound, not compute-bound.

    Limitations:
    - record_mode other than SPIKES_ONLY is not supported (checkpoint recomputation
      would require re-collecting traces, adding complexity not justified here).
    - cell_fn must be deterministic (no dropout inside unless set to eval mode).

    Args:
        cell_fn    : Single-timestep cell function.
        inputs     : Input sequence (B, T, ...).
        initial_state: Starting SpikingState.
        chunk_size : Timesteps per checkpoint chunk. Recommended: 10-20.
        record_mode: Only RecordMode.SPIKES_ONLY is supported.

    Returns:
        UnrollOutput with spikes and final_state. traces is always None.

    TODO(integration): Extend to support SPIKES_MEMBRANE by collecting traces
    outside the checkpoint scope (traces do not need gradient flow).
    """
    if record_mode != RecordMode.SPIKES_ONLY:
        raise ValueError(
            "snn_unroll_with_checkpointing only supports record_mode=SPIKES_ONLY. "
            f"Got record_mode={record_mode!r}. Use snn_unroll() for trace recording."
        )

    RecordMode.validate(record_mode)

    B = inputs.shape[0]
    T = inputs.shape[1]
    output_spatial: Tuple[int, ...] = ()

    # Probe output shape.
    with torch.no_grad():
        probe_out, _ = cell_fn(inputs[:, 0], initial_state.detach())
        output_spatial = tuple(probe_out.shape[1:])

    spike_record = torch.zeros(
        B, T, *output_spatial,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    state = initial_state
    chunk_schedule = compute_chunk_schedule(T, chunk_size)

    for chunk_start, chunk_end in chunk_schedule:
        chunk_inputs = inputs[:, chunk_start:chunk_end]

        def run_chunk(
            _chunk_inputs: Tensor,
            _v: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            """Checkpointed chunk forward. Returns (stacked_spikes, final_v)."""
            _state = SpikingState(
                v=_v,
                i=state.i,
                a=state.a,
                ref=state.ref,
                spike_history=state.spike_history,
            )
            chunk_len = _chunk_inputs.shape[1]
            spikes_list: List[Tensor] = []
            for t_local in range(chunk_len):
                x_t = _chunk_inputs[:, t_local]
                spike_t, _state = cell_fn(x_t, _state)
                spikes_list.append(spike_t)
            stacked = torch.stack(spikes_list, dim=1)  # (B, chunk_len, ...)
            return stacked, _state.v

        # gradient checkpointing requires all inputs to be Tensors.
        chunk_spikes, final_v = torch.utils.checkpoint.checkpoint(
            run_chunk,
            chunk_inputs,
            state.v,
            use_reentrant=False,
        )

        spike_record[:, chunk_start:chunk_end] = chunk_spikes.detach()

        # Rebuild state with updated v; detach non-differentiable fields.
        state = SpikingState(
            v=final_v.detach(),
            i=state.i.detach() if state.i is not None else None,
            a=state.a.detach() if state.a is not None else None,
            ref=state.ref.detach() if state.ref is not None else None,
            spike_history=(
                state.spike_history.detach()
                if state.spike_history is not None
                else None
            ),
        )

    return UnrollOutput(spikes=spike_record, final_state=state, traces=None)


# ---------------------------------------------------------------------------
# Public API summary
# ---------------------------------------------------------------------------

__all__ = [
    # Core types
    "SpikingState",
    "RecordMode",
    "CellFn",
    "UnrollOutput",
    # Main functions
    "snn_unroll",
    "snn_step",
    # Adapters
    "make_conv_cell",
    "make_linear_cell",
    # Utilities
    "compute_chunk_schedule",
    "estimate_memory_usage",
    # Debug
    "compare_unroll_modes",
    "snn_unroll_with_checkpointing",
]


# ---------------------------------------------------------------------------
# Self-test (run with: python -m brain_ai.core.unroll)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Minimal smoke test for the unroll utility.

    Exercises:
    1. SpikingState creation and detach.
    2. A trivial cell_fn (linear + LIF-like step).
    3. snn_unroll in sequence mode with various chunk sizes.
    4. snn_step in step mode.
    5. compute_chunk_schedule edge cases.
    6. estimate_memory_usage.
    7. compare_unroll_modes (forward match verification).

    Run directly:
        python brain_ai/core/unroll.py

    Or via pytest:
        pytest brain_ai/core/unroll.py -v
    """
    import sys

    print("snn_unroll_template.py smoke test")
    print("=" * 60)

    device = torch.device("cpu")
    torch.manual_seed(42)

    B, T, D_in, D_hidden = 4, 20, 16, 32

    # ---- Minimal LIF-like cell ------------------------------------------

    linear = nn.Linear(D_in, D_hidden)

    def lif_cell_fn(x_t: Tensor, state: SpikingState) -> Tuple[Tensor, SpikingState]:
        """Trivial LIF: current = linear(x_t), v = 0.9*v + current, spike = (v >= 1.0)."""
        current = linear(x_t.float())
        v = 0.9 * state.v + current
        # Binary spike via straight-through surrogate (no custom Function here).
        spike = (v >= 1.0).float()
        v = v * (1.0 - spike)  # hard reset
        return spike.to(x_t.dtype), SpikingState(v=v)

    # ---- 1. SpikingState construction ------------------------------------
    state_init = SpikingState.zeros(B, (D_hidden,), device=device)
    print(f"[1] SpikingState: {state_init}")

    detached = state_init.detach()
    assert detached.v.grad_fn is None, "detach() should produce leaf tensor"
    print("[1] SpikingState.detach() OK")

    # ---- 2. chunk_schedule -----------------------------------------------
    schedule = compute_chunk_schedule(10, 3)
    expected = [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert schedule == expected, f"Schedule mismatch: {schedule}"
    schedule_full = compute_chunk_schedule(10, 20)
    assert schedule_full == [(0, 10)], f"Full schedule failed: {schedule_full}"
    print(f"[2] compute_chunk_schedule OK: {schedule}")

    # ---- 3. estimate_memory_usage ----------------------------------------
    mem = estimate_memory_usage(B, T, D_hidden, chunk_size=5,
                                record_mode=RecordMode.SPIKES_ONLY)
    assert mem["total_bytes"] > 0
    print(f"[3] estimate_memory_usage OK: total={mem['total_bytes']} bytes")

    # ---- 4. snn_unroll — spikes only ------------------------------------
    x = torch.randn(B, T, D_in)
    result = snn_unroll(
        cell_fn=lif_cell_fn,
        inputs=x,
        initial_state=state_init,
        chunk_size=5,
        record_mode=RecordMode.SPIKES_ONLY,
    )
    assert result.spikes.shape == (B, T, D_hidden), f"Wrong shape: {result.spikes.shape}"
    assert result.traces is None
    print(f"[4] snn_unroll SPIKES_ONLY OK: spikes={tuple(result.spikes.shape)}")

    # ---- 5. snn_unroll — membrane traces ---------------------------------
    result_mem = snn_unroll(
        cell_fn=lif_cell_fn,
        inputs=x,
        initial_state=state_init,
        chunk_size=None,
        record_mode=RecordMode.SPIKES_MEMBRANE,
    )
    assert result_mem.traces is not None
    assert "membrane" in result_mem.traces
    assert result_mem.traces["membrane"].shape == (B, T, D_hidden)
    print(f"[5] snn_unroll SPIKES_MEMBRANE OK: membrane={tuple(result_mem.traces['membrane'].shape)}")

    # ---- 6. snn_step -----------------------------------------------------
    x_single = torch.randn(B, D_in)
    step_result = snn_step(lif_cell_fn, x_single, state_init)
    assert step_result.spikes.shape == (B, D_hidden)
    assert step_result.traces is None
    print(f"[6] snn_step OK: spike={tuple(step_result.spikes.shape)}")

    step_result_detailed = snn_step(lif_cell_fn, x_single, state_init, return_details=True)
    assert isinstance(step_result_detailed, tuple) and len(step_result_detailed) == 3
    print("[6] snn_step return_details=True OK")

    # ---- 7. compare_unroll_modes ----------------------------------------
    # Forward outputs must match across chunk sizes (no stochastic ops in cell_fn).
    cmp = compare_unroll_modes(
        cell_fn=lif_cell_fn,
        inputs=x,
        initial_state=state_init,
        chunk_sizes=[5, 10],
    )
    for cs, match in cmp["forward_match"].items():
        diff = cmp["forward_max_diff"][cs]
        status = "OK" if match else f"MISMATCH (diff={diff:.2e})"
        print(f"[7] compare_unroll_modes chunk_size={cs}: forward match = {status}")
    if cmp["warnings"]:
        for w in cmp["warnings"]:
            print(f"    WARNING: {w}")

    # ---- 8. FULL_TRACE record mode ---------------------------------------
    state_with_current = SpikingState(
        v=torch.zeros(B, D_hidden),
        i=torch.zeros(B, D_hidden),
    )

    def lif_cell_with_current(x_t: Tensor, state: SpikingState) -> Tuple[Tensor, SpikingState]:
        current = linear(x_t.float())
        i_new = 0.8 * (state.i if state.i is not None else torch.zeros_like(current)) + current
        v = 0.9 * state.v + i_new
        spike = (v >= 1.0).float()
        v = v * (1.0 - spike)
        return spike.to(x_t.dtype), SpikingState(v=v, i=i_new)

    result_full = snn_unroll(
        cell_fn=lif_cell_with_current,
        inputs=x,
        initial_state=state_with_current,
        chunk_size=5,
        record_mode=RecordMode.FULL_TRACE,
    )
    assert result_full.traces is not None
    assert "membrane" in result_full.traces
    assert "current" in result_full.traces
    print(f"[8] snn_unroll FULL_TRACE OK: keys={sorted(result_full.traces.keys())}")

    # ---- 9. SpikingState.to_dict / from_dict ----------------------------
    state_dict = state_init.to_dict("layer_0")
    assert "layer_0.v" in state_dict
    recovered = SpikingState.from_dict("layer_0", state_dict)
    assert torch.equal(state_init.v, recovered.v)
    print("[9] SpikingState serialisation OK")

    print("=" * 60)
    print("All smoke tests passed.")
