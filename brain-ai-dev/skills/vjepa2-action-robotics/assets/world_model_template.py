"""
World Model Template

Wraps a visual encoder (ViT) and an ActionConditionedPredictor into a
single inference-time WorldModel that exposes:
  - encode_image(image) -> representation
  - predict_next(repr, action, state) -> next representation
  - rollout(initial_repr, action_sequence, states) -> list of reprs
  - infer_action(current_image, goal_repr) -> action via CEM

This is the component used during MPC planning at inference time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    # Encoder settings
    encoder_embed_dim: int = 384      # ViT output dimension
    encoder_patch_size: int = 16      # Spatial patch size (pixels)
    image_size: int = 224             # Input image size (pixels)
    num_frames: int = 8               # Frames per clip during fine-tuning

    # Predictor settings
    predictor_embed_dim: int = 1024
    predictor_depth: int = 6          # Reduced depth for fast inference
    predictor_num_heads: int = 16
    use_extrinsics: bool = True

    # Inference settings
    device: str = "cpu"

    @property
    def num_patches(self) -> int:
        """Number of spatial patches per frame."""
        return (self.image_size // self.encoder_patch_size) ** 2

    @property
    def repr_dim(self) -> int:
        """Dimension of the visual representation (flattened patches)."""
        return self.num_patches * self.encoder_embed_dim


# ---------------------------------------------------------------------------
# Minimal Vision Transformer for testing
# ---------------------------------------------------------------------------

class MinimalViT(nn.Module):
    """
    Lightweight ViT encoder for testing and prototyping.
    In production, replace with the full V-JEPA 2 ViT encoder.
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 384, depth: int = 4, num_heads: int = 6,
                 in_channels: int = 3):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Minimal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            tokens: [B, num_patches, embed_dim]
        """
        x = self.patch_embed(x)                    # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)           # [B, num_patches, embed_dim]
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return x  # [B, num_patches, embed_dim]


# ---------------------------------------------------------------------------
# Simplified Predictor for single-step prediction
# ---------------------------------------------------------------------------

class SimplePredictorStep(nn.Module):
    """
    Lightweight predictor for single-step next-representation inference.
    Used inside WorldModel.predict_next() for fast rollout.

    In production, this should be replaced by ActionConditionedPredictor
    from ac_predictor_template.py.
    """

    def __init__(self, repr_dim: int, action_dim: int = 7, state_dim: int = 7,
                 hidden_dim: int = 512):
        super().__init__()
        # Concatenate repr + action + state -> predict next repr
        in_dim = repr_dim + action_dim + state_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, repr_dim)
        self.act = nn.GELU()
        # Residual connection: predict delta from current repr
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, repr_: Tensor, action: Tensor, state: Tensor) -> Tensor:
        """
        Args:
            repr_:  [B, repr_dim]
            action: [B, 7]
            state:  [B, 7]

        Returns:
            next_repr: [B, repr_dim]
        """
        combined = torch.cat([repr_, action, state], dim=-1)
        h = self.act(self.norm1(self.fc1(combined)))
        h = self.act(self.norm2(self.fc2(h)))
        delta = self.fc3(h) * self.residual_scale
        return repr_ + delta


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class WorldModel:
    """
    Inference wrapper combining encoder + predictor for MPC planning.

    Exposes the interface defined in SKILL.md:
        encode_image(image) -> Tensor
        predict_next(repr, action, state) -> Tensor
        rollout(initial_repr, action_sequence, states) -> List[Tensor]
        infer_action(current_image, goal_repr) -> Tensor
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        config: WorldModelConfig,
    ):
        """
        Args:
            encoder:   ViT encoder, forward(image) -> [B, N, D]
            predictor: AC predictor, forward(repr, action, state) -> repr
                       or SimplePredictorStep for fast rollout
            config:    WorldModelConfig
        """
        self.encoder = encoder
        self.predictor = predictor
        self.config = config
        self.device = torch.device(config.device)

        self.encoder.to(self.device)
        self.predictor.to(self.device)
        self.encoder.eval()
        self.predictor.eval()

    @torch.no_grad()
    def encode_image(self, image: Tensor) -> Tensor:
        """
        Encode a single frame or batch of frames.

        Args:
            image: [B, C, H, W] or [C, H, W] (single frame)

        Returns:
            repr: [B, num_patches, embed_dim] if encoder returns tokens
                  or [B, repr_dim] if encoder returns flat representation
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]

        image = image.to(self.device)
        tokens = self.encoder(image)  # [B, N, D] or [B, D]
        return tokens

    @torch.no_grad()
    def predict_next(
        self,
        current_repr: Tensor,
        action: Tensor,
        state: Tensor,
        extrinsics: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict the next representation given current state and action.

        Args:
            current_repr: [B, repr_dim] or [B, N, D] -- current representation
            action:       [B, 7] -- delta action
            state:        [B, 7] -- current robot state
            extrinsics:   [B, 6] optional -- camera extrinsics

        Returns:
            next_repr: same shape as current_repr
        """
        current_repr = current_repr.to(self.device)
        action = action.to(self.device)
        state = state.to(self.device)

        original_shape = current_repr.shape

        # Flatten to [B, repr_dim] if needed
        B = current_repr.shape[0]
        if current_repr.dim() > 2:
            repr_flat = current_repr.reshape(B, -1)
        else:
            repr_flat = current_repr

        next_repr_flat = self.predictor(repr_flat, action, state)

        # Restore original shape
        return next_repr_flat.reshape(original_shape)

    @torch.no_grad()
    def rollout(
        self,
        initial_repr: Tensor,
        action_sequence: Tensor,
        states: Tensor,
        extrinsics: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Autoregressive multi-step rollout through the world model.

        Args:
            initial_repr:    [B, repr_dim] or [B, N, D] -- starting representation
            action_sequence: [B, T, 7] or [T, 7] -- sequence of actions
            states:          [B, T, 7] or [T, 7] -- robot states at each step
                             (or [B, 7] for open-loop: state stays constant)
            extrinsics:      [B, T, 6] optional

        Returns:
            representations: List of T tensors, each [B, repr_dim]
                             (repr after applying each action)
        """
        initial_repr = initial_repr.to(self.device)
        action_sequence = action_sequence.to(self.device)
        states = states.to(self.device)

        # Normalize dimensions
        B = initial_repr.shape[0]

        if action_sequence.dim() == 2:
            action_sequence = action_sequence.unsqueeze(0).expand(B, -1, -1)
        if states.dim() == 1:
            states = states.unsqueeze(0).unsqueeze(0).expand(B, action_sequence.shape[1], -1)
        elif states.dim() == 2:
            if states.shape[0] == B:
                # [B, 7] -> open-loop: repeat for all timesteps
                states = states.unsqueeze(1).expand(B, action_sequence.shape[1], -1)
            else:
                # [T, 7] -> same state for all batch elements
                states = states.unsqueeze(0).expand(B, -1, -1)

        T = action_sequence.shape[1]
        repr_t = initial_repr
        representations = []

        for t in range(T):
            action_t = action_sequence[:, t, :]   # [B, 7]
            state_t  = states[:, t, :]             # [B, 7]

            extrin_t = None
            if extrinsics is not None:
                extrin_t = extrinsics[:, t, :].to(self.device)

            repr_t = self.predict_next(repr_t, action_t, state_t, extrin_t)
            representations.append(repr_t)

        return representations  # List[Tensor], len=T, each [B, repr_dim]

    @torch.no_grad()
    def infer_action(
        self,
        current_image: Tensor,
        goal_repr: Tensor,
        current_state: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Infer the next action to take using CEM planning.

        This is a thin wrapper that:
        1. Encodes current_image -> current_repr
        2. Runs CEMPlanner.plan(current_repr, goal_repr, current_state)
        3. Returns the planned action

        In full usage, construct a CEMPlanner separately and call plan() directly.
        This method provides a simple one-shot interface.

        Args:
            current_image: [B, C, H, W] or [C, H, W]
            goal_repr:     [repr_dim] or [B, repr_dim]
            current_state: [7] or [B, 7], defaults to zeros if None

        Returns:
            action: [7] -- first action of the planned sequence
        """
        try:
            from cem_planner_template import CEMPlanner, CEMConfig
        except ImportError:
            raise ImportError(
                "CEMPlanner not found. Ensure cem_planner_template.py is in the Python path."
            )

        if current_image.dim() == 3:
            current_image = current_image.unsqueeze(0)

        B = current_image.shape[0]
        current_repr = self.encode_image(current_image)  # [B, N, D] or [B, D]

        # Flatten representation
        if current_repr.dim() > 2:
            current_repr_flat = current_repr.reshape(B, -1)
        else:
            current_repr_flat = current_repr

        if goal_repr.dim() == 1:
            goal_repr = goal_repr.unsqueeze(0).expand(B, -1)

        if current_state is None:
            current_state = torch.zeros(B, 7, device=self.device)
        elif current_state.dim() == 1:
            current_state = current_state.unsqueeze(0).expand(B, -1)

        # CEM plans for single observation (use first batch element)
        repr_single  = current_repr_flat[0]   # [repr_dim]
        goal_single  = goal_repr[0]           # [repr_dim]
        state_single = current_state[0]       # [7]

        cem_config = CEMConfig()
        planner = CEMPlanner(self, cem_config)
        action = planner.plan(repr_single, goal_single, state_single)
        return action


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def create_world_model(config: Optional[WorldModelConfig] = None) -> WorldModel:
    """
    Create a WorldModel with minimal ViT encoder and simple predictor.
    For testing and prototyping. Replace encoder/predictor with full V-JEPA 2 components.
    """
    if config is None:
        config = WorldModelConfig(
            encoder_embed_dim=64,
            predictor_embed_dim=128,
            image_size=64,
            encoder_patch_size=16,
        )

    encoder = MinimalViT(
        image_size=config.image_size,
        patch_size=config.encoder_patch_size,
        embed_dim=config.encoder_embed_dim,
        depth=2,
        num_heads=4,
    )

    repr_dim = config.num_patches * config.encoder_embed_dim
    predictor = SimplePredictorStep(repr_dim=repr_dim, hidden_dim=256)

    return WorldModel(encoder=encoder, predictor=predictor, config=config)


def create_production_world_model(
    encoder: nn.Module,
    predictor: nn.Module,
    device: str = "cuda",
) -> WorldModel:
    """
    Create a WorldModel with production V-JEPA 2 encoder and AC predictor.

    Args:
        encoder:   Pre-loaded V-JEPA 2 ViT encoder
        predictor: ActionConditionedPredictor (from ac_predictor_template.py)
        device:    Target device

    Returns:
        WorldModel ready for inference
    """
    config = WorldModelConfig(device=device)
    return WorldModel(encoder=encoder, predictor=predictor, config=config)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_encode_image():
    """T95: encode_image returns correct shape."""
    print("[TEST] encode_image shape...")

    config = WorldModelConfig(
        encoder_embed_dim=64, image_size=64, encoder_patch_size=16,
        predictor_embed_dim=128, device="cpu"
    )
    wm = create_world_model(config)

    B = 3
    image = torch.randn(B, 3, 64, 64)
    repr_ = wm.encode_image(image)

    expected_N = config.num_patches
    assert repr_.shape == (B, expected_N, config.encoder_embed_dim), (
        f"encode_image shape: {repr_.shape}"
    )
    print(f"  encode_image shape: {repr_.shape} -- PASS")

    # Single frame (no batch dim)
    single_repr = wm.encode_image(torch.randn(3, 64, 64))
    assert single_repr.shape == (1, expected_N, config.encoder_embed_dim)
    print(f"  Single frame encode shape: {single_repr.shape} -- PASS")


def _test_rollout_length():
    """T94: rollout length matches action sequence length."""
    print("[TEST] rollout length matches action sequence...")

    config = WorldModelConfig(
        encoder_embed_dim=64, image_size=64, encoder_patch_size=16,
        predictor_embed_dim=128, device="cpu"
    )
    wm = create_world_model(config)

    B = 2
    repr_dim = config.num_patches * config.encoder_embed_dim
    initial_repr = torch.randn(B, repr_dim)

    for horizon in [1, 5, 10]:
        action_seq = torch.randn(B, horizon, 7)
        states = torch.randn(B, horizon, 7)
        reprs = wm.rollout(initial_repr, action_seq, states)

        assert len(reprs) == horizon, (
            f"Rollout length {len(reprs)} != horizon {horizon}"
        )
        assert reprs[0].shape == (B, repr_dim), (
            f"Repr shape {reprs[0].shape} != ({B}, {repr_dim})"
        )
        print(f"  Horizon={horizon}: {len(reprs)} steps, shape {reprs[0].shape} -- PASS")


def _test_predict_next_shape():
    """predict_next preserves representation shape."""
    print("[TEST] predict_next shape preservation...")

    config = WorldModelConfig(
        encoder_embed_dim=64, image_size=64, encoder_patch_size=16,
        predictor_embed_dim=128, device="cpu"
    )
    wm = create_world_model(config)

    B = 4
    repr_dim = config.num_patches * config.encoder_embed_dim
    repr_ = torch.randn(B, repr_dim)
    action = torch.randn(B, 7)
    state  = torch.randn(B, 7)

    next_repr = wm.predict_next(repr_, action, state)
    assert next_repr.shape == repr_.shape, (
        f"predict_next shape changed: {next_repr.shape} != {repr_.shape}"
    )
    print(f"  predict_next shape preserved: {next_repr.shape} -- PASS")


def _test_rollout_no_gradient():
    """Rollout runs under no_grad."""
    print("[TEST] rollout under no_grad...")

    config = WorldModelConfig(
        encoder_embed_dim=32, image_size=32, encoder_patch_size=16,
        device="cpu"
    )
    wm = create_world_model(config)

    repr_dim = config.num_patches * config.encoder_embed_dim
    initial = torch.randn(1, repr_dim)
    actions = torch.randn(1, 3, 7)
    states  = torch.randn(1, 3, 7)

    reprs = wm.rollout(initial, actions, states)
    for i, r in enumerate(reprs):
        assert not r.requires_grad, f"Repr {i} should not require grad"
    print(f"  No gradients in rollout output ({len(reprs)} steps) -- PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("WorldModel Self-Tests")
    print("=" * 60)

    _test_encode_image()
    _test_rollout_length()
    _test_predict_next_shape()
    _test_rollout_no_gradient()

    print("=" * 60)
    print("All self-tests PASSED")
    print("=" * 60)
