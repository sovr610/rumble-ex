# Pose Mathematics for Robot Action Computation

## Overview

V-JEPA 2-AC operates on delta actions (changes between consecutive poses) rather than
absolute poses. This document covers delta computation, delta application, camera frame
transforms, and the use of scipy.spatial.transform.Rotation for all euler<->matrix
conversions.

---

## Coordinate Conventions

### End-Effector Pose (7-DOF)

```
pose = [x, y, z, roll, pitch, yaw, gripper]
        m   m   m  rad   rad   rad  normalized
```

- **XYZ**: End-effector position in world frame (meters)
- **Roll/Pitch/Yaw**: Intrinsic Euler angles (xyz convention) in radians
- **Gripper**: Opening fraction, typically in range [-0.75, 0.75] in DROID

### Rotation Conventions

All rotation operations use scipy's `Rotation` class for numerical stability:

```python
from scipy.spatial.transform import Rotation as R

# Euler to rotation matrix
rot = R.from_euler('xyz', [roll, pitch, yaw])
mat = rot.as_matrix()  # [3, 3]

# Rotation matrix to Euler
rot = R.from_matrix(mat)
euler = rot.as_euler('xyz')  # [roll, pitch, yaw]

# Quaternion representation (for intermediate computations)
quat = rot.as_quat()  # [x, y, z, w]
```

---

## Delta Action Computation

### Pose-to-Delta

Given consecutive absolute poses, compute the delta action:

```python
def compute_delta_action(pose_t: np.ndarray, pose_tp1: np.ndarray) -> np.ndarray:
    """
    Compute delta action from pose_t to pose_tp1.

    Args:
        pose_t:   [7] -- current pose [x, y, z, roll, pitch, yaw, gripper]
        pose_tp1: [7] -- next pose

    Returns:
        delta: [7] -- [dx, dy, dz, droll, dpitch, dyaw, dgripper]
    """
    delta = np.zeros(7, dtype=np.float32)

    # Position delta: simple difference in world frame
    delta[0:3] = pose_tp1[0:3] - pose_t[0:3]

    # Orientation delta: rotation composition
    # delta_R = R_{t+1} * R_t^{-1}  (right-to-left composition)
    r_t   = R.from_euler('xyz', pose_t[3:6])
    r_tp1 = R.from_euler('xyz', pose_tp1[3:6])
    r_delta = r_tp1 * r_t.inv()
    delta[3:6] = r_delta.as_euler('xyz')

    # Gripper delta: simple difference
    delta[6] = pose_tp1[6] - pose_t[6]

    return delta
```

### Batch Delta Computation

```python
def compute_delta_sequence(poses: np.ndarray) -> np.ndarray:
    """
    Compute delta actions for a sequence of poses.

    Args:
        poses: [T, 7]

    Returns:
        deltas: [T-1, 7]
    """
    T = poses.shape[0]
    deltas = np.zeros((T - 1, 7), dtype=np.float32)
    for t in range(T - 1):
        deltas[t] = compute_delta_action(poses[t], poses[t + 1])
    return deltas
```

---

## Apply Delta Action

Given a current pose and a delta action, compute the next pose:

```python
def apply_delta_action(pose: np.ndarray, delta: np.ndarray,
                       gripper_clip: tuple = (-0.75, 0.75)) -> np.ndarray:
    """
    Apply delta action to current pose to get next pose.

    Args:
        pose:  [7] -- current [x, y, z, roll, pitch, yaw, gripper]
        delta: [7] -- [dx, dy, dz, droll, dpitch, dyaw, dgripper]

    Returns:
        next_pose: [7]
    """
    next_pose = np.zeros(7, dtype=np.float32)

    # Position: add translation delta
    next_pose[0:3] = pose[0:3] + delta[0:3]

    # Orientation: compose rotations
    # next_R = delta_R * R_t  (apply delta rotation to current)
    r_t     = R.from_euler('xyz', pose[3:6])
    r_delta = R.from_euler('xyz', delta[3:6])
    r_next  = r_delta * r_t
    next_pose[3:6] = r_next.as_euler('xyz')

    # Gripper: add delta and clip to valid range
    next_pose[6] = np.clip(pose[6] + delta[6], gripper_clip[0], gripper_clip[1])

    return next_pose
```

### Roundtrip Verification

Delta computation and application must be inverses of each other:

```python
def verify_pose_roundtrip(pose_t, pose_tp1, tol=1e-5):
    """Verify that compute_delta then apply_delta recovers pose_tp1."""
    delta = compute_delta_action(pose_t, pose_tp1)
    recovered = apply_delta_action(pose_t, delta)
    error = np.abs(recovered - pose_tp1).max()
    assert error < tol, f"Roundtrip error: {error} > {tol}"
    return error
```

Note: Euler angle roundtrips may have small numerical errors (~1e-7) due to gimbal lock
regions. Use quaternion representation if higher precision is needed.

---

## Camera Frame Transforms

### World Frame vs Camera Frame

DROID provides `camera_extrinsics [T, 4, 4]`, which is the **camera-to-world** transform
(i.e., it maps points from camera frame to world frame). To transform world-frame points
to camera frame, use the inverse:

```python
def world_to_camera(world_point: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """
    Transform a 3D point from world frame to camera frame.

    Args:
        world_point: [3] -- point in world frame
        extrinsics:  [4, 4] -- camera-to-world transform

    Returns:
        cam_point: [3]
    """
    # camera_T_world = inv(world_T_camera) = inv(extrinsics)
    camera_T_world = np.linalg.inv(extrinsics)

    # Homogeneous coordinates
    world_homog = np.append(world_point, 1.0)
    cam_homog = camera_T_world @ world_homog

    return cam_homog[:3]
```

### Pose in Camera Frame

```python
def world_pose_to_camera_frame(world_pose_mat: np.ndarray,
                                extrinsics: np.ndarray) -> np.ndarray:
    """
    Transform a 4x4 homogeneous pose from world frame to camera frame.

    Args:
        world_pose_mat: [4, 4] -- end-effector pose in world frame
        extrinsics:     [4, 4] -- camera-to-world transform

    Returns:
        camera_pose_mat: [4, 4]
    """
    # camera_T_world @ world_T_ee = camera_T_ee
    camera_T_world = np.linalg.inv(extrinsics)
    return camera_T_world @ world_pose_mat
```

### Build 4x4 Homogeneous Matrix from 7-DOF Pose

```python
def pose_to_homogeneous(pose: np.ndarray) -> np.ndarray:
    """
    Convert 7-DOF pose [x, y, z, roll, pitch, yaw, gripper] to 4x4 matrix.
    Gripper is not represented in the matrix (scalar DOF).

    Returns:
        H: [4, 4]
    """
    H = np.eye(4, dtype=np.float32)
    rot = R.from_euler('xyz', pose[3:6])
    H[:3, :3] = rot.as_matrix()
    H[:3, 3]  = pose[0:3]
    return H

def homogeneous_to_pose(H: np.ndarray, gripper: float = 0.0) -> np.ndarray:
    """
    Convert 4x4 homogeneous matrix back to 7-DOF pose.
    """
    pose = np.zeros(7, dtype=np.float32)
    pose[0:3] = H[:3, 3]
    rot = R.from_matrix(H[:3, :3])
    pose[3:6] = rot.as_euler('xyz')
    pose[6] = gripper
    return pose
```

---

## Euler Angle Edge Cases

### Gimbal Lock

When pitch = ±90 degrees, roll and yaw become degenerate (gimbal lock). This causes
numerical instability in euler-based delta computation. Mitigation:

```python
def safe_euler_diff(euler_t: np.ndarray, euler_tp1: np.ndarray) -> np.ndarray:
    """
    Compute rotation delta via rotation matrices to avoid gimbal lock issues.
    """
    r_t   = R.from_euler('xyz', euler_t)
    r_tp1 = R.from_euler('xyz', euler_tp1)
    r_delta = r_tp1 * r_t.inv()
    return r_delta.as_euler('xyz')
```

### Angle Wrapping

Euler angles wrap at ±pi. When computing deltas, unwrap angles:

```python
def unwrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    import math
    return (angle + math.pi) % (2 * math.pi) - math.pi
```

---

## Extrinsics Embedding for AC Predictor

Extract compact 6-DOF representation from full 4x4 extrinsics matrix:

```python
def extrinsics_to_6dof(extrinsics: np.ndarray) -> np.ndarray:
    """
    Extract 6-DOF embedding from 4x4 extrinsics matrix.
    Used as input to the AC predictor's extrinsics_embed layer.

    Args:
        extrinsics: [T, 4, 4] or [4, 4]

    Returns:
        dof6: [T, 6] or [6] -- [rx, ry, rz, tx, ty, tz]
    """
    single = extrinsics.ndim == 2
    if single:
        extrinsics = extrinsics[np.newaxis]  # [1, 4, 4]

    T = len(extrinsics)
    dof6 = np.zeros((T, 6), dtype=np.float32)

    for t in range(T):
        rot = R.from_matrix(extrinsics[t, :3, :3])
        dof6[t, 0:3] = rot.as_euler('xyz')  # rotation as euler
        dof6[t, 3:6] = extrinsics[t, :3, 3]  # translation

    return dof6[0] if single else dof6
```

---

## Summary of Key Relationships

```
pose_t [7] ---[compute_delta]--> delta [7]
                                    |
pose_t [7] ---[apply_delta]-------> pose_{t+1} [7]

world_pose [4,4] ---[inv(extrinsics) @]--> camera_pose [4,4]

euler [3] <-----> R.from_euler('xyz', euler) <-----> matrix [3,3]
euler [3] <-----> R.from_euler('xyz', euler) <-----> quat [4]
```

All operations use `scipy.spatial.transform.Rotation` for numerical stability and
correctness across all euler angle configurations.
