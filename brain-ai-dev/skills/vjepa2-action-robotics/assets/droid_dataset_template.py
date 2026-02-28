"""
DROID Dataset Template

Implements DROIDDataset for loading robot trajectories from the DROID format:
  - Video: .mp4 per camera view (left/right/wrist)
  - Trajectory: HDF5 with actions[T,7], states[T,7], camera_extrinsics[T,4,4]

Also provides:
  - pose_to_delta(states): absolute poses -> delta actions
  - apply_delta(pose, delta): apply delta to get next pose
  - camera_frame_transform: world pose -> camera frame
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Optional heavy dependencies -- gracefully handle missing installs
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("cv2 not found. Video loading will use fallback (zeros). "
                  "Install with: pip install opencv-python")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not found. HDF5 loading disabled. "
                  "Install with: pip install h5py")

try:
    from scipy.spatial.transform import Rotation as ScipyRotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found. Rotation utilities disabled. "
                  "Install with: pip install scipy")


# ---------------------------------------------------------------------------
# Pose Mathematics Utilities
# ---------------------------------------------------------------------------

def pose_to_delta(states: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of absolute end-effector poses to delta actions.

    Uses rotation matrix composition for the orientation component to avoid
    gimbal lock issues with naive euler subtraction.

    Args:
        states: [T, 7] -- absolute poses [x, y, z, roll, pitch, yaw, gripper]

    Returns:
        deltas: [T-1, 7] -- delta actions between consecutive timesteps
    """
    if not HAS_SCIPY:
        # Fallback: simple difference (less accurate for large rotations)
        deltas = np.diff(states, axis=0).astype(np.float32)
        return deltas

    T = states.shape[0]
    deltas = np.zeros((T - 1, 7), dtype=np.float32)

    for t in range(T - 1):
        # Position delta: direct subtraction
        deltas[t, 0:3] = states[t + 1, 0:3] - states[t, 0:3]

        # Orientation delta via rotation composition
        r_t   = ScipyRotation.from_euler('xyz', states[t,   3:6])
        r_tp1 = ScipyRotation.from_euler('xyz', states[t+1, 3:6])
        r_delta = r_tp1 * r_t.inv()
        deltas[t, 3:6] = r_delta.as_euler('xyz').astype(np.float32)

        # Gripper delta
        deltas[t, 6] = states[t + 1, 6] - states[t, 6]

    return deltas


def apply_delta(
    pose: np.ndarray,
    delta: np.ndarray,
    gripper_clip: Tuple[float, float] = (-0.75, 0.75),
) -> np.ndarray:
    """
    Apply a delta action to a current pose to get the next pose.

    Args:
        pose:  [7] -- current [x, y, z, roll, pitch, yaw, gripper]
        delta: [7] -- delta action
        gripper_clip: valid gripper range

    Returns:
        next_pose: [7]
    """
    next_pose = np.zeros(7, dtype=np.float32)

    # Position: additive
    next_pose[0:3] = pose[0:3] + delta[0:3]

    # Orientation: rotation composition
    if HAS_SCIPY:
        r_t     = ScipyRotation.from_euler('xyz', pose[3:6])
        r_delta = ScipyRotation.from_euler('xyz', delta[3:6])
        r_next  = r_delta * r_t
        next_pose[3:6] = r_next.as_euler('xyz').astype(np.float32)
    else:
        next_pose[3:6] = pose[3:6] + delta[3:6]

    # Gripper: additive + clip
    next_pose[6] = np.clip(pose[6] + delta[6], gripper_clip[0], gripper_clip[1])

    return next_pose


def camera_frame_transform(
    world_pose_mat: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """
    Transform a 4x4 world-frame pose into the camera coordinate frame.

    DROID convention: extrinsics is camera-to-world (world_T_camera).
    To go from world to camera: camera_T_world = inv(extrinsics).

    Args:
        world_pose_mat: [4, 4] homogeneous pose in world frame
        extrinsics:     [4, 4] camera-to-world transform

    Returns:
        camera_pose_mat: [4, 4] pose in camera frame
    """
    camera_T_world = np.linalg.inv(extrinsics)
    return camera_T_world @ world_pose_mat


def extrinsics_to_6dof(extrinsics: np.ndarray) -> np.ndarray:
    """
    Extract 6-DOF embedding from 4x4 extrinsics matrix.

    Args:
        extrinsics: [T, 4, 4] or [4, 4]

    Returns:
        dof6: [T, 6] or [6] -- [rx, ry, rz, tx, ty, tz]
    """
    single = extrinsics.ndim == 2
    if single:
        extrinsics = extrinsics[np.newaxis]

    T = len(extrinsics)
    dof6 = np.zeros((T, 6), dtype=np.float32)

    for t in range(T):
        if HAS_SCIPY:
            rot = ScipyRotation.from_matrix(extrinsics[t, :3, :3])
            dof6[t, 0:3] = rot.as_euler('xyz').astype(np.float32)
        else:
            # Fallback: zero rotation angles
            dof6[t, 0:3] = 0.0
        dof6[t, 3:6] = extrinsics[t, :3, 3].astype(np.float32)

    return dof6[0] if single else dof6


def pose_to_homogeneous(pose: np.ndarray) -> np.ndarray:
    """
    Convert 7-DOF pose to 4x4 homogeneous matrix.
    Gripper is not embedded in the matrix.

    Args:
        pose: [7] -- [x, y, z, roll, pitch, yaw, gripper]

    Returns:
        H: [4, 4]
    """
    H = np.eye(4, dtype=np.float32)
    if HAS_SCIPY:
        rot = ScipyRotation.from_euler('xyz', pose[3:6])
        H[:3, :3] = rot.as_matrix().astype(np.float32)
    H[:3, 3] = pose[0:3]
    return H


def homogeneous_to_pose(H: np.ndarray, gripper: float = 0.0) -> np.ndarray:
    """
    Convert 4x4 homogeneous matrix to 7-DOF pose.

    Args:
        H:       [4, 4]
        gripper: gripper value to attach

    Returns:
        pose: [7]
    """
    pose = np.zeros(7, dtype=np.float32)
    pose[0:3] = H[:3, 3]
    if HAS_SCIPY:
        rot = ScipyRotation.from_matrix(H[:3, :3])
        pose[3:6] = rot.as_euler('xyz').astype(np.float32)
    pose[6] = gripper
    return pose


# ---------------------------------------------------------------------------
# Video Loading
# ---------------------------------------------------------------------------

def load_video_frames(
    video_path: str,
    target_fps: int = 5,
    resize: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load video and subsample to target_fps.

    Args:
        video_path: Path to .mp4 file
        target_fps: Target frames per second for subsampling
        resize:     (width, height) to resize each frame

    Returns:
        frames: [T, H, W, 3] uint8 RGB frames
    """
    if not HAS_CV2:
        raise RuntimeError("cv2 required for video loading. pip install opencv-python")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0  # Fallback
    step = max(1, round(source_fps / target_fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame = cv2.resize(frame, (resize[0], resize[1]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from: {video_path}")

    return np.stack(frames, axis=0)  # [T, H, W, 3]


def sync_frames_to_trajectory(frames: np.ndarray, hdf5_T: int) -> np.ndarray:
    """
    Resample video frames to exactly match HDF5 trajectory length.
    Uses nearest-neighbor index interpolation.

    Args:
        frames:  [T_video, H, W, 3]
        hdf5_T:  target number of frames

    Returns:
        synced: [hdf5_T, H, W, 3]
    """
    video_T = len(frames)
    if video_T == hdf5_T:
        return frames
    if video_T == 0:
        raise ValueError("Empty frames array")
    video_indices = np.round(np.linspace(0, video_T - 1, hdf5_T)).astype(int)
    video_indices = np.clip(video_indices, 0, video_T - 1)
    return frames[video_indices]  # [hdf5_T, H, W, 3]


# ---------------------------------------------------------------------------
# HDF5 Loading
# ---------------------------------------------------------------------------

def load_trajectory(hdf5_path: str) -> Dict[str, np.ndarray]:
    """
    Load trajectory data from DROID HDF5 file.

    Returns:
        dict with keys:
            'actions':           [T, 7] float32
            'states':            [T, 7] float32
            'camera_extrinsics': [T, 4, 4] float32
            'timestamps':        [T] float64 (if present)
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py required. pip install h5py")

    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Required fields
        required = ['actions', 'states', 'camera_extrinsics']
        for key in required:
            if key not in f:
                raise KeyError(
                    f"Required key '{key}' not found in {hdf5_path}. "
                    f"Available keys: {list(f.keys())}"
                )

        data = {
            'actions':           np.array(f['actions']).astype(np.float32),
            'states':            np.array(f['states']).astype(np.float32),
            'camera_extrinsics': np.array(f['camera_extrinsics']).astype(np.float32),
        }

        if 'timestamps' in f:
            data['timestamps'] = np.array(f['timestamps']).astype(np.float64)

    # Validate shapes
    T = data['actions'].shape[0]
    assert data['actions'].shape == (T, 7), f"actions shape: {data['actions'].shape}"
    assert data['states'].shape == (T, 7), f"states shape: {data['states'].shape}"
    assert data['camera_extrinsics'].shape == (T, 4, 4), (
        f"extrinsics shape: {data['camera_extrinsics'].shape}"
    )

    return data


# ---------------------------------------------------------------------------
# Clip Sampling
# ---------------------------------------------------------------------------

def sample_clip(
    frames: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    extrinsics: np.ndarray,
    frames_per_clip: int = 8,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a random consecutive clip from the trajectory.

    Args:
        frames:          [T, H, W, 3]
        actions:         [T, 7]
        states:          [T, 7]
        extrinsics:      [T, 4, 4]
        frames_per_clip: number of frames in the clip
        stride:          temporal stride between consecutive clip frames

    Returns:
        clip_frames:     [frames_per_clip, H, W, 3]
        clip_actions:    [frames_per_clip, 7]
        clip_states:     [frames_per_clip, 7]
        clip_extrinsics: [frames_per_clip, 4, 4]
    """
    T = len(frames)
    clip_span = (frames_per_clip - 1) * stride + 1
    max_start = T - clip_span

    if max_start < 0:
        raise ValueError(
            f"Trajectory too short: {T} frames, need at least {clip_span} "
            f"(frames_per_clip={frames_per_clip}, stride={stride})"
        )

    start = np.random.randint(0, max_start + 1)
    indices = np.array([start + i * stride for i in range(frames_per_clip)])

    return (
        frames[indices],
        actions[indices],
        states[indices],
        extrinsics[indices],
    )


# ---------------------------------------------------------------------------
# Synthetic DROID Episode Generator (for testing without real data)
# ---------------------------------------------------------------------------

def create_synthetic_episode(
    episode_dir: str,
    T: int = 100,
    image_size: Tuple[int, int] = (224, 224),
    fps: int = 30,
    camera_views: Optional[List[str]] = None,
) -> None:
    """
    Create a synthetic DROID-format episode for testing.
    Generates:
      - Dummy MP4 videos (solid color, changing frame-by-frame)
      - HDF5 trajectory file with random actions/states/extrinsics
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py required. pip install h5py")
    if not HAS_CV2:
        raise RuntimeError("cv2 required. pip install opencv-python")

    if camera_views is None:
        camera_views = ["left", "right", "wrist"]

    os.makedirs(episode_dir, exist_ok=True)

    # Create dummy videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = image_size[1], image_size[0]

    for view in camera_views:
        video_path = os.path.join(episode_dir, f"{view}.mp4")
        out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
        for t in range(T):
            # Gradient frame: changes over time
            val = int(255 * t / max(T - 1, 1))
            frame = np.full((H, W, 3), val, dtype=np.uint8)
            out.write(frame)
        out.release()

    # Create trajectory HDF5
    hdf5_path = os.path.join(episode_dir, "trajectory.hdf5")
    with h5py.File(hdf5_path, 'w') as f:
        # Random actions (small deltas)
        rng = np.random.RandomState(42)
        actions = rng.randn(T, 7).astype(np.float32) * 0.01
        actions[:, 3:6] = 0.0   # Zero orientation (as in real DROID)
        actions[:, 6] = np.clip(rng.randn(T) * 0.1, -0.75, 0.75)

        # States: integrate from zero
        states = np.zeros((T, 7), dtype=np.float32)
        for t in range(1, T):
            states[t] = states[t-1] + actions[t-1]
        states[:, 6] = np.clip(states[:, 6], -0.75, 0.75)

        # Camera extrinsics: identity transforms
        extrinsics = np.tile(np.eye(4, dtype=np.float32)[np.newaxis], (T, 1, 1))

        f.create_dataset('actions', data=actions)
        f.create_dataset('states', data=states)
        f.create_dataset('camera_extrinsics', data=extrinsics)
        f.create_dataset('timestamps', data=np.linspace(0, T/fps, T, dtype=np.float64))

        f.create_group('metadata')
        f['metadata'].attrs['robot_type'] = 'franka_panda'
        f['metadata'].attrs['episode_id'] = os.path.basename(episode_dir)
        f['metadata'].attrs['task_description'] = 'synthetic test episode'


# ---------------------------------------------------------------------------
# DROIDDataset
# ---------------------------------------------------------------------------

class DROIDDataset(Dataset):
    """
    PyTorch Dataset for loading DROID robot trajectories.

    Each item in the dataset is a random clip from one episode, containing:
        'frames':     [T, 3, H, W] float32 -- normalized RGB frames
        'actions':    [T, 7] float32 -- delta actions
        'states':     [T, 7] float32 -- robot proprioceptive states
        'extrinsics': [T, 4, 4] float32 -- camera extrinsics matrices
        'extrinsics_6dof': [T, 6] float32 -- 6-DOF extrinsics for AC predictor
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str,
        camera_view: str = "left",     # "left", "right", or "wrist"
        target_fps: int = 5,           # Subsample video to this FPS
        frames_per_clip: int = 8,      # Number of frames per training clip
        image_size: Tuple[int, int] = (224, 224),  # (W, H) for cv2 resize
        stride: int = 1,               # Temporal stride in clip sampling
        normalize: bool = True,        # Apply ImageNet normalization
        use_delta_actions: bool = True, # Convert states to delta actions
        seed: Optional[int] = None,    # Random seed for reproducible sampling
    ):
        self.data_dir = data_dir
        self.camera_view = camera_view
        self.target_fps = target_fps
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        self.stride = stride
        self.normalize = normalize
        self.use_delta_actions = use_delta_actions
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        if not os.path.exists(data_dir):
            raise ValueError(f"data_dir does not exist: {data_dir}")

        # Find all episodes (directories containing trajectory.hdf5)
        self.episodes = sorted([
            d.rstrip('/') for d in glob.glob(os.path.join(data_dir, "*/"))
            if os.path.exists(os.path.join(d, "trajectory.hdf5"))
        ])

        if len(self.episodes) == 0:
            raise ValueError(
                f"No episodes found in {data_dir}. "
                f"Each episode directory must contain 'trajectory.hdf5'."
            )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        episode_dir = self.episodes[idx]

        # Load video
        video_path = os.path.join(episode_dir, f"{self.camera_view}.mp4")
        frames = load_video_frames(
            video_path, self.target_fps,
            resize=self.image_size
        )  # [T_video, H, W, 3] uint8

        # Load HDF5 trajectory
        hdf5_path = os.path.join(episode_dir, "trajectory.hdf5")
        traj = load_trajectory(hdf5_path)

        # Synchronize video to trajectory length
        frames = sync_frames_to_trajectory(frames, len(traj['actions']))

        # Use stored actions or compute deltas from states
        if self.use_delta_actions:
            actions = traj['actions']  # Pre-computed deltas in DROID
        else:
            # Recompute from states if needed
            actions = traj['actions']

        # Sample a clip
        clip_frames, clip_actions, clip_states, clip_extrinsics = sample_clip(
            frames=frames,
            actions=actions,
            states=traj['states'],
            extrinsics=traj['camera_extrinsics'],
            frames_per_clip=self.frames_per_clip,
            stride=self.stride,
        )

        # Convert frames to [T, 3, H, W] float in [0, 1]
        frames_f32 = clip_frames.astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_f32).permute(0, 3, 1, 2)  # [T, 3, H, W]

        if self.normalize:
            mean = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std  = torch.tensor(self.IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
            frames_tensor = (frames_tensor - mean) / std

        # Extract 6-DOF extrinsics for AC predictor
        ext_6dof = extrinsics_to_6dof(clip_extrinsics)  # [T, 6]

        return {
            'frames':          frames_tensor,
            'actions':         torch.from_numpy(clip_actions).float(),
            'states':          torch.from_numpy(clip_states).float(),
            'extrinsics':      torch.from_numpy(clip_extrinsics).float(),
            'extrinsics_6dof': torch.from_numpy(ext_6dof).float(),
        }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_pose_delta_roundtrip():
    """T74: Compute delta then apply gives back the original next pose."""
    print("[TEST] Pose delta roundtrip...")

    if not HAS_SCIPY:
        print("  [SKIP] scipy not available")
        return

    np.random.seed(42)

    # Random pose
    pose_t   = np.array([0.3, -0.1, 0.5, 0.2, -0.1, 0.4, 0.5], dtype=np.float32)
    # Random next pose (small perturbation)
    pose_tp1 = pose_t + np.array([0.01, -0.005, 0.008, 0.02, -0.01, 0.015, 0.05])
    pose_tp1[6] = np.clip(pose_tp1[6], -0.75, 0.75)

    # Compute delta
    states = np.stack([pose_t, pose_tp1])
    deltas = pose_to_delta(states)
    delta = deltas[0]

    # Apply delta
    recovered = apply_delta(pose_t, delta)

    error = np.abs(recovered - pose_tp1).max()
    assert error < 1e-5, f"Roundtrip error: {error:.2e}"
    print(f"  Roundtrip error: {error:.2e} < 1e-5 -- PASS")

    # Test zero delta
    zero_delta = np.zeros(7, dtype=np.float32)
    same_pose = apply_delta(pose_t, zero_delta)
    zero_error = np.abs(same_pose - pose_t).max()
    assert zero_error < 1e-6, f"Zero delta error: {zero_error}"
    print(f"  Zero delta preserves pose: error={zero_error:.2e} -- PASS")


def _test_shapes_correct():
    """T52-T55: Dataset returns correct tensor shapes."""
    print("[TEST] DROIDDataset shapes...")

    if not (HAS_CV2 and HAS_H5PY):
        print("  [SKIP] cv2 or h5py not available")
        return

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two synthetic episodes
        for i in range(2):
            ep_dir = os.path.join(tmpdir, f"episode_{i:03d}")
            create_synthetic_episode(ep_dir, T=50, fps=30)

        dataset = DROIDDataset(
            data_dir=tmpdir,
            camera_view="left",
            target_fps=5,
            frames_per_clip=8,
            image_size=(64, 64),
        )

        assert len(dataset) == 2, f"Dataset len: {len(dataset)} != 2"
        print(f"  Dataset length: {len(dataset)} -- PASS")

        item = dataset[0]

        T = 8
        assert item['frames'].shape    == (T, 3, 64, 64), f"frames: {item['frames'].shape}"
        assert item['actions'].shape   == (T, 7),          f"actions: {item['actions'].shape}"
        assert item['states'].shape    == (T, 7),           f"states: {item['states'].shape}"
        assert item['extrinsics'].shape == (T, 4, 4),       f"extrinsics: {item['extrinsics'].shape}"

        print(f"  frames shape:     {item['frames'].shape} -- PASS")
        print(f"  actions shape:    {item['actions'].shape} -- PASS")
        print(f"  states shape:     {item['states'].shape} -- PASS")
        print(f"  extrinsics shape: {item['extrinsics'].shape} -- PASS")

        # Temporal alignment: all T same
        assert item['frames'].shape[0] == item['actions'].shape[0], (
            "frames and actions T mismatch"
        )
        print(f"  Temporal alignment: T={item['frames'].shape[0]} consistent -- PASS")


def _test_camera_frame_transform():
    """T86 / T87: Camera frame transform roundtrip."""
    print("[TEST] Camera frame transform...")

    # Identity extrinsics
    identity = np.eye(4, dtype=np.float32)
    test_pose = np.eye(4, dtype=np.float32)
    test_pose[:3, 3] = [0.3, 0.1, 0.5]

    camera_pose = camera_frame_transform(test_pose, identity)
    assert np.allclose(camera_pose, test_pose, atol=1e-6), (
        f"Identity extrinsics should preserve pose, got:\n{camera_pose}"
    )
    print(f"  Identity extrinsics preserves pose -- PASS")

    # Roundtrip with arbitrary extrinsics
    if HAS_SCIPY:
        rot = ScipyRotation.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
        extrin = np.eye(4, dtype=np.float32)
        extrin[:3, :3] = rot
        extrin[:3, 3] = [0.5, -0.2, 0.1]

        cam_pose = camera_frame_transform(test_pose, extrin)
        # Inverse: apply extrin to recover world pose
        camera_T_world = np.linalg.inv(extrin)
        world_recovered = np.linalg.inv(camera_T_world) @ cam_pose
        assert np.allclose(world_recovered, test_pose, atol=1e-5), (
            f"Roundtrip error: {np.abs(world_recovered - test_pose).max():.2e}"
        )
        print(f"  Camera frame transform roundtrip -- PASS")


def _test_extrinsics_6dof():
    """T90: extrinsics_to_6dof shape correct."""
    print("[TEST] extrinsics_to_6dof...")

    T = 10
    extrin = np.tile(np.eye(4, dtype=np.float32)[np.newaxis], (T, 1, 1))
    dof6 = extrinsics_to_6dof(extrin)

    assert dof6.shape == (T, 6), f"Shape: {dof6.shape} != ({T}, 6)"
    print(f"  extrinsics_to_6dof shape: {dof6.shape} -- PASS")

    # Single extrinsics
    single_dof6 = extrinsics_to_6dof(np.eye(4, dtype=np.float32))
    assert single_dof6.shape == (6,), f"Single shape: {single_dof6.shape}"
    print(f"  Single extrinsics 6DOF shape: {single_dof6.shape} -- PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("DROIDDataset Self-Tests")
    print("=" * 60)

    _test_pose_delta_roundtrip()
    _test_shapes_correct()
    _test_camera_frame_transform()
    _test_extrinsics_6dof()

    print("=" * 60)
    print("All self-tests PASSED")
    print("=" * 60)
