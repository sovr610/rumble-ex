# Testing Matrix: V-JEPA 2 Action-Conditioned & Robotics

## Overview

This document enumerates all test scenarios for the V-JEPA 2-AC skill. Tests are organized
by component and categorized by the done-when gate they validate.

---

## Done-When Gate 1: AC Forward Shape & Mask

Tests that `ActionConditionedPredictor.forward()` produces correct output shape with
interleaved tokens, and that the block-causal mask prevents future-frame attention.

### AC Predictor Shape Tests

| Test ID | Description | Input | Expected Output |
|---|---|---|---|
| `T01` | Basic forward shape, no extrinsics | B=2, T=4, N=49, D=384 | `[B, T*N, predictor_dim]` |
| `T02` | Basic forward shape, with extrinsics | B=2, T=4, N=49, D=384 | `[B, T*N, predictor_dim]` |
| `T03` | Single frame (T=1) | B=1, T=1, N=16, D=256 | `[B, N, predictor_dim]` |
| `T04` | Large batch (B=16) | B=16, T=8, N=196, D=1024 | `[B, T*N, predictor_dim]` |
| `T05` | Minimum config (minimal predictor) | B=1, T=2, N=4, D=64 | `[B, T*N, predictor_dim]` |
| `T06` | Output dtype matches input dtype | float32 input | float32 output |
| `T07` | No NaN/Inf in output | random inputs | all finite |
| `T08` | Gradient flows through all parameters | backward pass | all params.grad not None |

### Token Interleaving Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T09` | Total token count without extrinsics | `total = T * (2 + N)` |
| `T10` | Total token count with extrinsics | `total = T * (3 + N)` |
| `T11` | action_token_mask has correct True count | `mask.sum() == T * 2` (no extrinsics) |
| `T12` | action_token_mask True count with extrinsics | `mask.sum() == T * 3` |
| `T13` | First token in each frame block is state token | mask[0], mask[K], mask[2K], ... all True |
| `T14` | Last N tokens in each frame block are visual | mask[K-N:K], mask[2K-N:2K], ... all False |
| `T15` | Interleaving is deterministic (same input -> same output) | Two calls identical |

### Block-Causal Mask Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T16` | Mask shape | `[T*K, T*K]` where K=tokens_per_frame |
| `T17` | Frame 0 cannot attend to frame 1 | `mask[0:K, K:2K].isinf()` all True |
| `T18` | Frame 1 can attend to frame 0 | `mask[K:2K, 0:K] == 0` all True |
| `T19` | Diagonal blocks are zero (within-frame bidirectional) | `mask[t*K:(t+1)*K, t*K:(t+1)*K] == 0` |
| `T20` | Upper triangle blocks are -inf | all upper-diagonal K×K blocks are -inf |
| `T21` | Lower triangle blocks are 0 | all lower-diagonal K×K blocks are 0 |
| `T22` | With pred_is_frame_causal=False, mask is all zeros | no -inf entries |
| `T23` | Mask does not have -inf on main diagonal | diagonal entries are 0 |

### AC-RoPE Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T24` | Action tokens get different RoPE than visual tokens | `q_action != q_visual` for same position |
| `T25` | Visual tokens have non-zero height/width RoPE rotation | height/width components of visual q differ from unrotated |
| `T26` | Action tokens have zero height/width RoPE rotation | height/width freq components masked to 0 for action tokens |
| `T27` | Output shape unchanged after AC-RoPE | same shape in, same shape out |
| `T28` | ACBlock produces correct output shape | `[B, total_tokens, dim]` |
| `T29` | ACBlock gradient flows | backward pass succeeds |

---

## Done-When Gate 2: CEM Convergence

Tests that `CEMPlanner.plan()` returns an action reducing L1 distance to goal over iterations.

### CEM Core Algorithm Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T30` | plan() returns shape [7] | output shape == (7,) |
| `T31` | CEM reduces cost over iterations (synthetic model) | `cost_iter_K < cost_iter_0` |
| `T32` | CEM convergence: final cost < 0.5 * initial cost | at least 50% improvement |
| `T33` | Elite selection: top-k by cost | `costs[elites].max() <= costs[non_elites].min()` |
| `T34` | mu update applies momentum correctly | `new_mu != elite_mean` when momentum > 0 |
| `T35` | sigma never reaches 0 (clamp at 1e-6) | `sigma.min() >= 1e-6` after K iterations |
| `T36` | Plan is deterministic for fixed seed | same plan with same torch.manual_seed |

### Action Clipping Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T37` | XYZ components clipped to ±maxnorm | `abs(action[0:3]) <= maxnorm` |
| `T38` | Orientation (3:6) zeroed | `action[3:6] == 0` |
| `T39` | Gripper clipped to [-0.75, 0.75] | `action[6] in [-0.75, 0.75]` |
| `T40` | Clipping handles already-valid actions (no-op) | small valid action unchanged |
| `T41` | Clipping handles extreme values | very large action clipped correctly |
| `T42` | Clipping is applied before scoring | unclipped sequences never evaluated |

### CEM Convergence Tests (Synthetic World Model)

| Test ID | Description | Model | Expected |
|---|---|---|---|
| `T43` | Identity world model: goal = current | repr -> repr | action near zero |
| `T44` | Linear world model: simple goal-reaching | repr -> repr + action[:repr_dim] | action close to optimal |
| `T45` | Cost monotonically decreases over K iterations | any model | `costs[k+1] <= costs[k]` |
| `T46` | Larger num_samples gives better final cost | same goal, N=512 vs N=64 | N=512 cost <= N=64 cost |
| `T47` | Sigma collapses from 1.0 to near 0 for easy problem | identity-like model | sigma decreases |

### Batched Rollout Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T48` | Batched rollout shape | `[N, repr_dim]` |
| `T49` | Batched rollout identical to sequential rollout | max diff < 1e-5 |
| `T50` | Rollout with horizon=1 | single predict_next call |
| `T51` | Rollout with horizon=10 | 10 predict_next calls per sequence |

---

## Done-When Gate 3: DROID Loading

Tests that `DROIDDataset` loads synchronized video frames + actions + states with correct
shapes and temporal alignment.

### DROIDDataset Loading Tests

| Test ID | Description | Expected |
|---|---|---|
| `T52` | frames shape | `[T, 3, H, W]` with T=frames_per_clip |
| `T53` | actions shape | `[T, 7]` |
| `T54` | states shape | `[T, 7]` |
| `T55` | extrinsics shape | `[T, 4, 4]` |
| `T56` | frames are normalized (ImageNet) | mean close to 0, std close to 1 |
| `T57` | actions and frames have same T | `frames.shape[0] == actions.shape[0]` |
| `T58` | Dataset length matches number of episodes | `len(dataset) == num_episodes` |
| `T59` | Repeated __getitem__ gives different clips | two calls differ (random start) |

### Video Loading Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T60` | load_video_frames returns correct shape | `[T_sub, H, W, 3]` |
| `T61` | FPS subsampling correct | `T_sub ≈ video_duration * target_fps` |
| `T62` | Frame values in [0, 255] uint8 | before normalization |
| `T63` | No dropped frames (all frames present) | len(frames) > 0 |
| `T64` | Handles missing video file gracefully | FileNotFoundError raised |

### HDF5 Loading Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T65` | actions dtype float32 | `actions.dtype == float32` |
| `T66` | states dtype float32 | `states.dtype == float32` |
| `T67` | extrinsics is valid SE(3) matrix | det(R) ≈ 1.0 for each timestep |
| `T68` | Handles missing HDF5 keys gracefully | appropriate error raised |
| `T69` | No NaN values in loaded data | all finite |

### Temporal Synchronization Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T70` | sync_frames_to_trajectory preserves endpoint | first and last frame match original |
| `T71` | Synchronized frames match HDF5 length | `len(synced) == hdf5_T` |
| `T72` | Subsampled video aligns with actions | temporal correlation preserved |

---

## Pose Mathematics Tests

### Delta Computation Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T73` | Delta of identical poses is zero | `compute_delta_action(p, p) ≈ 0` |
| `T74` | Delta roundtrip: apply(p, delta(p, p')) == p' | max error < 1e-5 |
| `T75` | Position delta is exact subtraction | `delta[0:3] == p_tp1[0:3] - p_t[0:3]` |
| `T76` | Gripper delta is exact subtraction | `delta[6] == p_tp1[6] - p_t[6]` |
| `T77` | Orientation delta composes correctly | rotation matrix identity test |
| `T78` | Batch delta sequence has shape [T-1, 7] | `compute_delta_sequence(poses).shape == (T-1, 7)` |
| `T79` | Large rotation delta (near pi) handled | no NaN, result finite |
| `T80` | Gimbal lock region (pitch=90 deg) handled | no NaN, result finite |

### Apply Delta Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T81` | apply_delta(p, zeros) == p | identity delta preserves pose |
| `T82` | Gripper clamped after apply_delta | `result[6] in [-0.75, 0.75]` |
| `T83` | Pure translation delta (zero rotation) | xyz changes, euler unchanged |
| `T84` | Pure rotation delta (zero translation) | xyz unchanged, euler changes |
| `T85` | Sequential apply_delta matches compute_delta | start/end poses match |

### Camera Transform Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T86` | world_to_camera roundtrip | `camera_to_world(world_to_camera(p)) ≈ p` |
| `T87` | Identity extrinsics preserves point | `world_to_camera(p, I_4x4) ≈ p` |
| `T88` | pose_to_homogeneous output is valid SE(3) | `det(H[:3,:3]) ≈ 1.0` |
| `T89` | homogeneous_to_pose roundtrip | `homogeneous_to_pose(pose_to_homogeneous(p)) ≈ p` |
| `T90` | extrinsics_to_6dof shape correct | `[T, 6]` for `[T, 4, 4]` input |

---

## Integration Tests

| Test ID | Description | Assertion |
|---|---|---|
| `T91` | Full pipeline: DROID load -> encode -> predict | no errors, output shape correct |
| `T92` | CEM planning on DROID observation | action shape [7], values clipped |
| `T93` | AC predictor on DROID clip | output shape `[B, T*N, predictor_dim]` |
| `T94` | WorldModel.rollout length matches horizon | `len(rollout) == horizon` |
| `T95` | WorldModel.encode_image output shape | `[B, N, D]` or `[B, D]` |
| `T96` | WorldModel.infer_action returns [7] tensor | correct shape and clipping |
| `T97` | AC predictor with T=1 (single frame inference) | shape `[B, N, predictor_dim]` |
| `T98` | CEM with horizon=1 | valid action returned |
| `T99` | Full MPC loop: 5 plan-execute steps | no errors, robot moves toward goal |

---

## Edge Case Tests

| Test ID | Description | Expected Behavior |
|---|---|---|
| `T100` | Zero-length action sequence to CEM | raises ValueError |
| `T101` | Goal == current state | CEM returns near-zero action |
| `T102` | All-zero actions in CEM sample | handled gracefully |
| `T103` | Empty DROID dataset directory | raises ValueError |
| `T104` | Single-frame video | handled gracefully, frames_per_clip must be 1 |
| `T105` | Very short trajectory (T < frames_per_clip) | raises ValueError or pads |

---

## Performance Benchmarks (non-blocking)

| Test ID | Description | Target |
|---|---|---|
| `P01` | AC predictor forward (B=4, T=8) | < 100ms on CPU |
| `P02` | CEM plan (N=512, K=5, horizon=10) | < 2s on CPU |
| `P03` | DROID dataset __getitem__ | < 500ms per sample |
| `P04` | WorldModel.rollout (horizon=10) | < 200ms on GPU |
