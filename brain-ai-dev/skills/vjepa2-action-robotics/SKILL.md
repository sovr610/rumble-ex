---
name: V-JEPA 2 Action-Conditioned & Robotics
description: >
  This skill should be used when the user asks to "implement V-JEPA 2-AC",
  "action-conditioned world model", "block-causal attention",
  "token interleaving for actions", "DROID dataset integration",
  "model predictive control", "CEM planning", "robot pose computation",
  "autoregressive rollout", "frame-causal prediction",
  "AC-RoPE implementation", "action embedding", "7-DOF robot control",
  "world model wrapper", "cross-entropy method optimizer",
  or needs guidance on action-conditioned prediction, robotics planning,
  pose mathematics, or DROID fine-tuning for V-JEPA 2.
version: 0.1.0
---

# V-JEPA 2 Action-Conditioned & Robotics

## Overview

Guide implementation of V-JEPA 2-AC, the action-conditioned variant for robotics. Cover the AC predictor architecture with token interleaving, block-causal attention masking, AC-RoPE (action tokens get only depth-axis rotation), autoregressive rollout, DROID dataset integration (video + HDF5 trajectory), the WorldModel inference wrapper, and the Cross-Entropy Method (CEM) planner for Model Predictive Control.

## Public Contract

### ActionConditionedPredictor

Predictor with action/state/extrinsics conditioning.

```python
class ActionConditionedPredictor(nn.Module):
    def __init__(self, embed_dim, predictor_embed_dim, depth, num_heads,
                 action_embed_dim=7, state_embed_dim=7, use_extrinsics=True,
                 pred_is_frame_causal=True): ...
    def forward(self, context_repr: Tensor, actions: Tensor, states: Tensor,
                extrinsics: Optional[Tensor] = None) -> Tensor: ...
```

### ACRoPEAttention

RoPE attention with special action token handling.

```python
class ACRoPEAttention(nn.Module):
    def __init__(self, dim, num_heads): ...
    def forward(self, x: Tensor, rope_freqs: Tensor,
                causal_mask: Optional[Tensor] = None,
                action_token_mask: Tensor = None) -> Tensor: ...
```

### WorldModel

Inference wrapper combining encoder + predictor for MPC planning.

```python
class WorldModel:
    def __init__(self, encoder: VisionTransformer,
                 predictor: ActionConditionedPredictor, config: WorldModelConfig): ...
    def encode_image(self, image: Tensor) -> Tensor: ...
    def predict_next(self, current_repr: Tensor, action: Tensor,
                     state: Tensor) -> Tensor: ...
    def rollout(self, initial_repr: Tensor, action_sequence: Tensor,
                states: Tensor) -> List[Tensor]: ...
    def infer_action(self, current_image: Tensor, goal_repr: Tensor) -> Tensor: ...
```

### CEMPlanner

Cross-Entropy Method optimizer for action planning.

```python
class CEMPlanner:
    def __init__(self, world_model: WorldModel, config: CEMConfig): ...
    def plan(self, current_repr: Tensor, goal_repr: Tensor,
             current_state: Tensor) -> Tensor: ...
```

### DROIDDataset

Dataset loading robot trajectories from DROID format.

```python
class DROIDDataset(Dataset):
    def __init__(self, data_dir: str, camera_view: str = "left",
                 target_fps: int = 5, frames_per_clip: int = 8): ...
    def __getitem__(self, idx) -> Dict[str, Tensor]: ...
```

## Key Concepts

### Token Interleaving Structure

Per frame in the AC predictor:
```
[state_token, (extrinsics_token), action_token, visual_token_1, ..., visual_token_HW]
```
- State: robot proprioceptive state (7-DOF)
- Extrinsics: camera pose (6-DOF, optional)
- Action: delta action command (7-DOF)
- Visual: encoder output tokens for that frame

### Block-Causal Attention Mask

Each frame's tokens can attend to current and all previous frames' tokens:
```
Frame 0: attends to [Frame 0]
Frame 1: attends to [Frame 0, Frame 1]
Frame 2: attends to [Frame 0, Frame 1, Frame 2]
```
Enables autoregressive next-frame prediction while allowing within-frame bidirectional attention.

### AC-RoPE

- Visual tokens: full 3-axis RoPE (depth, height, width)
- Action/state/extrinsics tokens: only depth-axis rotation (no spatial position)
- Detected via `action_token_mask` boolean tensor

### Autoregressive Rollout

`auto_steps` controls multi-step prediction:
1. Predict next frame representation from current context + action
2. Feed prediction back as input for next step
3. Loss computed at each step and averaged
4. Enables multi-step planning without ground truth intermediate frames

### CEM Planning Algorithm

```
Initialize: mu = zeros(horizon, 7), sigma = ones(horizon, 7)
Repeat K iterations:
  1. Sample N action sequences ~ N(mu, sigma^2)
  2. Clip: xyz to ±maxnorm, gripper to [-0.75, 0.75], orientation = 0
  3. Roll out each through world model to get final representations
  4. Score: L1(final_repr, goal_repr)
  5. Select top-k (elites)
  6. mu = momentum * mu + (1-momentum) * mean(elites)
     sigma = momentum * sigma + (1-momentum) * std(elites)
Return mu[0] as optimal first action
```

### DROID Dataset Format

- Video: `.mp4` files per camera view (left, right, wrist)
- Trajectory: HDF5 with `actions [T, 7]`, `states [T, 7]`, `camera_extrinsics [T, 4, 4]`
- 7-DOF: `[x, y, z, roll, pitch, yaw, gripper]`
- Pose-to-diff: absolute poses -> delta actions via rotation matrix differences

### Pose Mathematics

- **Delta action**: `action = pose_{t+1} - pose_t` (position) + rotation matrix difference (orientation)
- **Apply delta**: New xyz + composed rotation + clipped gripper from current pose + action
- **Camera frame transform**: `world_pose -> camera_frame` via `extrinsics^{-1} @ pose`
- Uses `scipy.spatial.transform.Rotation` for euler<->matrix conversions

## Configuration Surface

```python
@dataclass
class ACPredictorConfig:
    predictor_embed_dim: int = 1024
    predictor_depth: int = 24
    predictor_num_heads: int = 16
    action_embed_dim: int = 7
    state_embed_dim: int = 7
    use_extrinsics: bool = True
    pred_is_frame_causal: bool = True

@dataclass
class CEMConfig:
    horizon: int = 10
    num_samples: int = 512
    num_elites: int = 64
    num_iterations: int = 5
    momentum_xyz: float = 0.1
    momentum_gripper: float = 0.3
    maxnorm: float = 0.02
    gripper_range: Tuple[float, float] = (-0.75, 0.75)
```

## Done-When Gates

1. **AC Forward** — `ActionConditionedPredictor.forward()` produces correct output shape with interleaved tokens; block-causal mask prevents future-frame attention.
2. **CEM Planning** — `CEMPlanner.plan()` returns action that reduces L1 distance to goal over iterations on synthetic world model.
3. **DROID Loading** — `DROIDDataset` loads synchronized video frames + actions + states with correct shapes and temporal alignment.

## Resources

### Reference Files
- **`references/ac-predictor.md`** — Token interleaving, block-causal mask, AC-RoPE
- **`references/cem-planning.md`** — CEM algorithm, momentum updates, action clipping
- **`references/droid-dataset.md`** — DROID format, pose processing, camera transforms
- **`references/pose-mathematics.md`** — Delta actions, rotation composition, coordinate frames
- **`references/testing-matrix.md`** — Test scenarios

### Asset Files
- **`assets/ac_predictor_template.py`** — ActionConditionedPredictor with interleaving, self-tests
- **`assets/ac_rope_template.py`** — ACRoPEAttention, ACBlock with action token handling
- **`assets/world_model_template.py`** — WorldModel wrapper with encode, predict, rollout
- **`assets/cem_planner_template.py`** — CEMPlanner with momentum and clipping
- **`assets/droid_dataset_template.py`** — DROIDDataset with pose processing utilities

### Scripts
- **`scripts/validate_ac.py`** — Validates done-when gates
- **`scripts/gen_ac_tests.py`** — Generates 100+ pytest test cases
- **`scripts/mpc_demo.py`** — Demo of CEM planning with synthetic world model
