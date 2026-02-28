"""
world_model_template.py
=======================
Template for BaseWorldModel — the composition root of the world model scaffold.

BaseWorldModel is an nn.Module that holds component instances (encoder, dynamics,
decoder, and optionally memory and planner) injected at construction time.
Dimensional compatibility is validated in __init__ using each component's
dimension-reporting methods.

Usage (self-test):
    python world_model_template.py
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Import the ABC base classes.
# In the generated scaffold these are imported from the package structure.
# When running this file standalone they are imported from base_classes_template.
# ---------------------------------------------------------------------------
try:
    from world_model.encoders.base import BaseEncoder
    from world_model.dynamics.base import BaseDynamics
    from world_model.memory.base import BaseMemory
    from world_model.planning.base import BasePlanner
    from world_model.decoders.base import BaseDecoder
except ImportError:
    # Fallback for standalone execution / scaffold template rendering
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from base_classes_template import (
        BaseEncoder,
        BaseDynamics,
        BaseMemory,
        BasePlanner,
        BaseDecoder,
    )


# ---------------------------------------------------------------------------
# BaseWorldModel
# ---------------------------------------------------------------------------


class BaseWorldModel(nn.Module):
    """Composition root for a world model.

    Holds encoder, dynamics, decoder, and optional memory and planner as
    nn.Module children.  Delegates all computation to these components.

    Components are injected at construction — BaseWorldModel does not
    instantiate them internally and has no knowledge of their concrete types.
    This enables hot-swap: pass a different encoder instance to get a
    different model, with zero changes to dynamics, decoder, or this class.

    Dimensional validation
    ----------------------
    At construction time, the following invariants are checked::

        encoder.get_embed_dim() == dynamics.get_state_dim()
        encoder.get_embed_dim() == decoder.get_input_dim()

    Violation raises ValueError immediately, before any forward pass.

    Forward pipeline
    ----------------
    encode(obs)        : obs (B,*) -> latent (B, embed_dim)
    step(state, action): latent (B, D) x action (B, A) -> next_state (B, D)
    decode(latent)     : latent (B, D) -> obs_hat (B, *output_shape)
    forward(obs, action): full pipeline in one call

    Parameters
    ----------
    encoder : BaseEncoder
        Maps observations to latent vectors.
    dynamics : BaseDynamics
        Predicts next latent state from (state, action).
    decoder : BaseDecoder
        Reconstructs observations from latent vectors.
    memory : Optional[BaseMemory]
        Episodic memory.  When provided, step() reads context from memory
        before calling dynamics and writes the new state back.
    planner : Optional[BasePlanner]
        Action planner.  When provided, accessible via self.planner.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        dynamics: BaseDynamics,
        decoder: BaseDecoder,
        memory: Optional[BaseMemory] = None,
        planner: Optional[BasePlanner] = None,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # Type checks — fail fast with clear messages
        # ---------------------------------------------------------------
        if not isinstance(encoder, BaseEncoder):
            raise TypeError(
                f"encoder must be a BaseEncoder instance, got {type(encoder).__name__}"
            )
        if not isinstance(dynamics, BaseDynamics):
            raise TypeError(
                f"dynamics must be a BaseDynamics instance, got {type(dynamics).__name__}"
            )
        if not isinstance(decoder, BaseDecoder):
            raise TypeError(
                f"decoder must be a BaseDecoder instance, got {type(decoder).__name__}"
            )
        if memory is not None and not isinstance(memory, BaseMemory):
            raise TypeError(
                f"memory must be a BaseMemory instance or None, got {type(memory).__name__}"
            )
        if planner is not None and not isinstance(planner, BasePlanner):
            raise TypeError(
                f"planner must be a BasePlanner instance or None, "
                f"got {type(planner).__name__}"
            )

        # ---------------------------------------------------------------
        # Dimensional validation — checked once at construction time
        # ---------------------------------------------------------------
        enc_dim = encoder.get_embed_dim()
        dyn_dim = dynamics.get_state_dim()
        dec_dim = decoder.get_input_dim()

        if enc_dim <= 0:
            raise ValueError(
                f"encoder.get_embed_dim() must be > 0, got {enc_dim}"
            )
        if enc_dim != dyn_dim:
            raise ValueError(
                f"Dimension mismatch: encoder.get_embed_dim()={enc_dim} != "
                f"dynamics.get_state_dim()={dyn_dim}. "
                f"Both must be equal for correct composition."
            )
        if enc_dim != dec_dim:
            raise ValueError(
                f"Dimension mismatch: encoder.get_embed_dim()={enc_dim} != "
                f"decoder.get_input_dim()={dec_dim}. "
                f"Both must be equal for correct composition."
            )

        # ---------------------------------------------------------------
        # Register as nn.Module children so .parameters(), .to(), etc. work
        # ---------------------------------------------------------------
        # Concrete encoder/dynamics/decoder must also inherit nn.Module
        # for parameter registration to work.  ABCs do not require this,
        # but production implementations should.
        self.encoder = encoder  # type: ignore[assignment]
        self.dynamics = dynamics  # type: ignore[assignment]
        self.decoder = decoder  # type: ignore[assignment]

        self._has_memory = memory is not None
        self._has_planner = planner is not None

        if memory is not None:
            self.memory = memory  # type: ignore[assignment]
        if planner is not None:
            self.planner = planner  # type: ignore[assignment]

        # Store dimension for external inspection
        self._embed_dim = enc_dim

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        """Embedding dimension shared by encoder, dynamics, and decoder."""
        return self._embed_dim

    @property
    def state_dim(self) -> int:
        """Latent state dimension (alias for embed_dim)."""
        return self._embed_dim

    @property
    def has_memory(self) -> bool:
        """Return True if this model has a memory component."""
        return self._has_memory

    @property
    def has_planner(self) -> bool:
        """Return True if this model has a planner component."""
        return self._has_planner

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """The shape of one decoded observation (no batch dimension)."""
        return self.decoder.get_output_shape()

    # -----------------------------------------------------------------------
    # Core methods — each delegates to one component
    # -----------------------------------------------------------------------

    def encode(self, obs: Tensor) -> Tensor:
        """Encode a batch of observations into latent embeddings.

        Parameters
        ----------
        obs : Tensor
            Shape (B, *obs_shape).

        Returns
        -------
        Tensor
            Shape (B, embed_dim).
        """
        return self.encoder.forward(obs)

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the next latent state given the current state and action.

        If memory is present, read context from memory before calling dynamics,
        then write the new state to memory.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).
        action : Tensor
            Shape (B, action_dim).

        Returns
        -------
        Tensor
            Shape (B, state_dim).
        """
        if self._has_memory:
            context = self.memory.read(state)
            # Add context to state; concrete subclasses may override
            # this fusion strategy (concatenation, cross-attention, etc.)
            augmented_state = state + context
            next_state = self.dynamics.step(augmented_state, action)
            self.memory.write(state, next_state)
        else:
            next_state = self.dynamics.step(state, action)
        return next_state

    def imagine(
        self, state: Tensor, policy: Callable[[Tensor], Tensor], horizon: int
    ) -> Tensor:
        """Roll out an imagined trajectory using the dynamics model.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).  Initial latent state.
        policy : Callable[[Tensor], Tensor]
            Maps (B, state_dim) -> (B, action_dim).
        horizon : int
            Number of steps.  Must be >= 1.

        Returns
        -------
        Tensor
            Shape (B, horizon, state_dim).
        """
        return self.dynamics.imagine(state, policy, horizon)

    def decode(self, latent: Tensor) -> Tensor:
        """Decode a batch of latent vectors to reconstructed observations.

        Parameters
        ----------
        latent : Tensor
            Shape (B, embed_dim).

        Returns
        -------
        Tensor
            Shape (B, *output_shape).
        """
        return self.decoder.forward(latent)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Run the full encode -> step -> decode pipeline.

        Parameters
        ----------
        obs : Tensor
            Shape (B, *obs_shape).  Current observation.
        action : Tensor
            Shape (B, action_dim).  Action taken.

        Returns
        -------
        Tensor
            Shape (B, *output_shape).  Reconstructed next observation.
        """
        latent = self.encode(obs)
        next_state = self.step(latent, action)
        return self.decode(next_state)

    # -----------------------------------------------------------------------
    # Extra utilities
    # -----------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Reset the memory component if present, otherwise no-op."""
        if self._has_memory:
            self.memory.reset()

    def plan(self, state: Tensor, horizon: int) -> Tensor:
        """Plan an action sequence using the planner component.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).  Current latent state.
        horizon : int
            Number of steps to plan.

        Returns
        -------
        Tensor
            Shape (B, horizon, action_dim).

        Raises
        ------
        RuntimeError
            If this model has no planner component.
        """
        if not self._has_planner:
            raise RuntimeError(
                "plan() called on a BaseWorldModel without a planner. "
                "Pass a BasePlanner instance at construction time."
            )
        return self.planner.plan(state, self.dynamics, horizon)

    def __repr__(self) -> str:
        lines = [
            f"BaseWorldModel(",
            f"  embed_dim={self._embed_dim}",
            f"  encoder={type(self.encoder).__name__}",
            f"  dynamics={type(self.dynamics).__name__}",
            f"  decoder={type(self.decoder).__name__}",
            f"  memory={type(self.memory).__name__ if self._has_memory else None}",
            f"  planner={type(self.planner).__name__ if self._has_planner else None}",
            f")",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test block
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    errors: list[str] = []

    def fail(msg: str) -> None:
        errors.append(f"FAIL: {msg}")

    def ok(msg: str) -> None:
        print(f"  PASS: {msg}")

    # -----------------------------------------------------------------------
    # Minimal concrete implementations for testing
    # -----------------------------------------------------------------------

    class _MockEncoder(nn.Module, BaseEncoder):
        def __init__(self, embed_dim: int = 64):
            super().__init__()
            self._embed_dim = embed_dim
            self.proj = nn.Linear(32, embed_dim)

        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros(x.shape[0], self._embed_dim)

        def get_embed_dim(self) -> int:
            return self._embed_dim

        def get_output_shape(self):
            return (self._embed_dim,)

    class _MockDynamics(nn.Module, BaseDynamics):
        def __init__(self, state_dim: int = 64, action_dim: int = 4):
            super().__init__()
            self._state_dim = state_dim
            self.net = nn.Linear(state_dim + action_dim, state_dim)

        def step(self, state: Tensor, action: Tensor) -> Tensor:
            return torch.zeros_like(state)

        def imagine(self, state: Tensor, policy: Callable, horizon: int) -> Tensor:
            B = state.shape[0]
            return torch.zeros(B, horizon, self._state_dim)

        def get_state_dim(self) -> int:
            return self._state_dim

    class _MockDecoder(nn.Module, BaseDecoder):
        def __init__(self, input_dim: int = 64):
            super().__init__()
            self._input_dim = input_dim
            self.net = nn.Linear(input_dim, 3 * 16 * 16)

        def forward(self, latent: Tensor) -> Tensor:
            return torch.zeros(latent.shape[0], 3, 16, 16)

        def get_output_shape(self):
            return (3, 16, 16)

        def get_input_dim(self) -> int:
            return self._input_dim

    class _MockMemory(nn.Module, BaseMemory):
        def __init__(self, dim: int = 64):
            super().__init__()
            self._dim = dim

        def read(self, query: Tensor) -> Tensor:
            return torch.zeros(query.shape[0], self._dim)

        def write(self, key: Tensor, value: Tensor) -> None:
            pass

        def reset(self) -> None:
            pass

        def get_capacity(self) -> int:
            return 128

    class _MockPlanner(BasePlanner):
        def plan(self, state: Tensor, dynamics: BaseDynamics, horizon: int) -> Tensor:
            return torch.zeros(state.shape[0], horizon, 4)

    # -----------------------------------------------------------------------
    # Test 1: Basic construction without optional components
    # -----------------------------------------------------------------------
    print("=== Test 1: Basic construction (no memory, no planner) ===")
    try:
        model = BaseWorldModel(
            encoder=_MockEncoder(64),
            dynamics=_MockDynamics(64, 4),
            decoder=_MockDecoder(64),
        )
        ok("BaseWorldModel constructs without error")
        assert not model.has_memory, "has_memory must be False"
        assert not model.has_planner, "has_planner must be False"
        assert model.embed_dim == 64, "embed_dim must be 64"
        ok("has_memory=False, has_planner=False, embed_dim=64")
    except Exception as exc:
        fail(f"Basic construction failed: {exc}")

    # -----------------------------------------------------------------------
    # Test 2: Dimensional mismatch — encoder vs dynamics
    # -----------------------------------------------------------------------
    print("\n=== Test 2: Dimensional mismatch (encoder != dynamics) ===")
    try:
        BaseWorldModel(
            encoder=_MockEncoder(64),
            dynamics=_MockDynamics(128, 4),
            decoder=_MockDecoder(64),
        )
        fail("Expected ValueError for encoder/dynamics mismatch, got no error")
    except ValueError as exc:
        ok(f"ValueError raised: {exc}")
    except Exception as exc:
        fail(f"Wrong exception type {type(exc).__name__}: {exc}")

    # -----------------------------------------------------------------------
    # Test 3: Dimensional mismatch — encoder vs decoder
    # -----------------------------------------------------------------------
    print("\n=== Test 3: Dimensional mismatch (encoder != decoder) ===")
    try:
        BaseWorldModel(
            encoder=_MockEncoder(64),
            dynamics=_MockDynamics(64, 4),
            decoder=_MockDecoder(128),
        )
        fail("Expected ValueError for encoder/decoder mismatch, got no error")
    except ValueError as exc:
        ok(f"ValueError raised: {exc}")
    except Exception as exc:
        fail(f"Wrong exception type {type(exc).__name__}: {exc}")

    # -----------------------------------------------------------------------
    # Test 4: Wrong type passed for encoder
    # -----------------------------------------------------------------------
    print("\n=== Test 4: Wrong type for encoder ===")
    try:
        BaseWorldModel(
            encoder="not_an_encoder",  # type: ignore
            dynamics=_MockDynamics(64, 4),
            decoder=_MockDecoder(64),
        )
        fail("Expected TypeError for wrong encoder type")
    except TypeError as exc:
        ok(f"TypeError raised: {exc}")

    # -----------------------------------------------------------------------
    # Test 5: Forward pass shapes
    # -----------------------------------------------------------------------
    print("\n=== Test 5: Forward pass shapes ===")
    try:
        model = BaseWorldModel(
            encoder=_MockEncoder(64),
            dynamics=_MockDynamics(64, 4),
            decoder=_MockDecoder(64),
        )
        B = 3
        obs = torch.randn(B, 32)
        action = torch.randn(B, 4)

        latent = model.encode(obs)
        assert latent.shape == (B, 64), f"encode shape mismatch: {latent.shape}"
        ok(f"encode: {latent.shape}")

        next_state = model.step(latent, action)
        assert next_state.shape == (B, 64), f"step shape mismatch: {next_state.shape}"
        ok(f"step: {next_state.shape}")

        recon = model.decode(latent)
        assert recon.shape == (B, 3, 16, 16), f"decode shape mismatch: {recon.shape}"
        ok(f"decode: {recon.shape}")

        out = model.forward(obs, action)
        assert out.shape == (B, 3, 16, 16), f"forward shape mismatch: {out.shape}"
        ok(f"forward: {out.shape}")

        traj = model.imagine(latent, lambda s: torch.zeros(B, 4), horizon=5)
        assert traj.shape == (B, 5, 64), f"imagine shape mismatch: {traj.shape}"
        ok(f"imagine: {traj.shape}")

    except Exception as exc:
        fail(f"Forward pass shapes test failed: {exc}")

    # -----------------------------------------------------------------------
    # Test 6: With memory and planner
    # -----------------------------------------------------------------------
    print("\n=== Test 6: With memory and planner ===")
    try:
        model_full = BaseWorldModel(
            encoder=_MockEncoder(64),
            dynamics=_MockDynamics(64, 4),
            decoder=_MockDecoder(64),
            memory=_MockMemory(64),
            planner=_MockPlanner(),
        )
        assert model_full.has_memory, "has_memory must be True"
        assert model_full.has_planner, "has_planner must be True"
        ok("Full model with memory and planner constructs")

        B = 2
        obs = torch.randn(B, 32)
        action = torch.randn(B, 4)
        out = model_full.forward(obs, action)
        assert out.shape == (B, 3, 16, 16), f"full forward shape mismatch: {out.shape}"
        ok(f"Full forward pass: {out.shape}")

        plan = model_full.plan(torch.randn(B, 64), horizon=4)
        assert plan.shape == (B, 4, 4), f"plan shape mismatch: {plan.shape}"
        ok(f"Plan: {plan.shape}")

        model_full.reset_memory()
        ok("reset_memory() succeeded")

    except Exception as exc:
        fail(f"Full model test failed: {exc}")

    # -----------------------------------------------------------------------
    # Test 7: Hot-swap — replace encoder A with encoder B
    # -----------------------------------------------------------------------
    print("\n=== Test 7: Hot-swap encoder ===")
    try:
        dyn = _MockDynamics(64, 4)
        dec = _MockDecoder(64)
        model_a = BaseWorldModel(
            encoder=_MockEncoder(64), dynamics=dyn, decoder=dec
        )
        model_b = BaseWorldModel(
            encoder=_MockEncoder(64), dynamics=dyn, decoder=dec
        )
        B = 2
        obs = torch.randn(B, 32)
        action = torch.randn(B, 4)
        out_a = model_a.forward(obs, action)
        out_b = model_b.forward(obs, action)
        assert out_a.shape == out_b.shape, "Hot-swap output shapes must match"
        ok("Hot-swap: both models produce same-shaped outputs")
    except Exception as exc:
        fail(f"Hot-swap test failed: {exc}")

    # -----------------------------------------------------------------------
    # Test 8: plan() raises RuntimeError when no planner
    # -----------------------------------------------------------------------
    print("\n=== Test 8: plan() without planner raises RuntimeError ===")
    try:
        model_no_planner = BaseWorldModel(
            encoder=_MockEncoder(64),
            dynamics=_MockDynamics(64, 4),
            decoder=_MockDecoder(64),
        )
        model_no_planner.plan(torch.randn(2, 64), horizon=3)
        fail("Expected RuntimeError when calling plan() without planner")
    except RuntimeError as exc:
        ok(f"RuntimeError raised: {exc}")
    except Exception as exc:
        fail(f"Wrong exception type: {type(exc).__name__}: {exc}")

    # -----------------------------------------------------------------------
    # Test 9: repr
    # -----------------------------------------------------------------------
    print("\n=== Test 9: __repr__ ===")
    try:
        model = BaseWorldModel(
            encoder=_MockEncoder(64),
            dynamics=_MockDynamics(64, 4),
            decoder=_MockDecoder(64),
        )
        r = repr(model)
        assert "BaseWorldModel" in r, "__repr__ must contain 'BaseWorldModel'"
        assert "embed_dim=64" in r, "__repr__ must contain embed_dim"
        ok(f"__repr__ works: first line = '{r.splitlines()[0]}'")
    except Exception as exc:
        fail(f"__repr__ test failed: {exc}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    if errors:
        for e in errors:
            print(f"  {e}")
        print(f"\n{len(errors)} test(s) FAILED.")
        sys.exit(1)
    else:
        print("All BaseWorldModel tests PASSED.")
        sys.exit(0)
