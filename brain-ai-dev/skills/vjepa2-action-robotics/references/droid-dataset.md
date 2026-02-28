# DROID Dataset Format and Integration

## Overview

DROID (Diverse Robot Open-source Demonstrations) is a large-scale robot manipulation dataset
used to fine-tune V-JEPA 2-AC. Each trajectory consists of synchronized video from multiple
cameras and a trajectory HDF5 file with actions, states, and camera extrinsics.

---

## File Structure

### Per-Trajectory Layout

```
<episode_id>/
    left.mp4          -- Left stereo camera video
    right.mp4         -- Right stereo camera video
    wrist.mp4         -- Wrist-mounted camera video
    trajectory.hdf5   -- Robot trajectory data
```

### HDF5 Schema

```
trajectory.hdf5
├── actions           [T, 7]      float32  -- Delta action commands
├── states            [T, 7]      float32  -- Robot proprioceptive states
├── camera_extrinsics [T, 4, 4]   float32  -- World-to-camera transforms
├── timestamps        [T]         float64  -- Unix timestamps (seconds)
└── metadata/
    ├── robot_type    str
    ├── episode_id    str
    └── task_description str
```

### 7-DOF Action/State Vector

```
[0] x         -- End-effector x position (meters)
[1] y         -- End-effector y position (meters)
[2] z         -- End-effector z position (meters)
[3] roll      -- End-effector roll (radians)
[4] pitch     -- End-effector pitch (radians)
[5] yaw       -- End-effector yaw (radians)
[6] gripper   -- Gripper opening (normalized, -1=closed, +1=open; DROID uses ~[-0.75, 0.75])
```

---

## Video Loading

### Frame Extraction

DROID videos are typically recorded at 30 Hz. For V-JEPA 2-AC fine-tuning, subsample to
target_fps (typically 5-15 Hz) to match the action rate.

```python
import cv2
import numpy as np

def load_video_frames(video_path: str, target_fps: int = 5,
                      resize: tuple = (224, 224)) -> np.ndarray:
    """
    Load video and subsample to target_fps.

    Returns:
        frames: [T_subsampled, H, W, 3] uint8
    """
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(source_fps / target_fps))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return np.stack(frames, axis=0)  # [T, H, W, 3]
```

### Temporal Synchronization

Video frames must be aligned with HDF5 trajectory steps:

```python
def sync_frames_to_trajectory(frames: np.ndarray, hdf5_T: int) -> np.ndarray:
    """
    Resample video frames to exactly match HDF5 trajectory length.
    Uses linear interpolation of frame indices.
    """
    video_T = len(frames)
    if video_T == hdf5_T:
        return frames
    # Linear resampling
    video_indices = np.linspace(0, video_T - 1, hdf5_T).astype(int)
    return frames[video_indices]  # [hdf5_T, H, W, 3]
```

---

## Loading Trajectory Data

```python
import h5py

def load_trajectory(hdf5_path: str) -> dict:
    """
    Load all trajectory data from HDF5.

    Returns dict with keys:
        actions:           [T, 7]    float32
        states:            [T, 7]    float32
        camera_extrinsics: [T, 4, 4] float32
        timestamps:        [T]       float64 (if present)
    """
    with h5py.File(hdf5_path, 'r') as f:
        data = {
            'actions':           np.array(f['actions']),           # [T, 7]
            'states':            np.array(f['states']),            # [T, 7]
            'camera_extrinsics': np.array(f['camera_extrinsics']), # [T, 4, 4]
        }
        if 'timestamps' in f:
            data['timestamps'] = np.array(f['timestamps'])
    return data
```

---

## Clip Sampling

V-JEPA 2-AC is trained on short clips of `frames_per_clip` consecutive frames.

```python
def sample_clip(frames, actions, states, extrinsics,
                frames_per_clip: int = 8, stride: int = 1):
    """
    Sample a random clip of length frames_per_clip from a trajectory.

    Returns:
        clip_frames:     [T, H, W, 3]
        clip_actions:    [T, 7]
        clip_states:     [T, 7]
        clip_extrinsics: [T, 4, 4]
    """
    T = len(frames)
    max_start = T - frames_per_clip * stride
    assert max_start >= 0, f"Trajectory too short: {T} frames < {frames_per_clip}"

    start = np.random.randint(0, max_start + 1)
    indices = [start + i * stride for i in range(frames_per_clip)]

    return (
        frames[indices],       # [T, H, W, 3]
        actions[indices],      # [T, 7]
        states[indices],       # [T, 7]
        extrinsics[indices],   # [T, 4, 4]
    )
```

---

## DROIDDataset: Complete Interface

```python
from torch.utils.data import Dataset
import torch
import os, glob

class DROIDDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        camera_view: str = "left",    # "left", "right", or "wrist"
        target_fps: int = 5,
        frames_per_clip: int = 8,
        image_size: tuple = (224, 224),
        stride: int = 1,
        normalize: bool = True,
    ):
        self.data_dir = data_dir
        self.camera_view = camera_view
        self.target_fps = target_fps
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        self.stride = stride
        self.normalize = normalize

        # Find all episodes (directories with trajectory.hdf5)
        self.episodes = sorted([
            d for d in glob.glob(os.path.join(data_dir, "*/"))
            if os.path.exists(os.path.join(d, "trajectory.hdf5"))
        ])

        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found in {data_dir}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx) -> dict:
        episode_dir = self.episodes[idx]

        # Load video
        video_path = os.path.join(episode_dir, f"{self.camera_view}.mp4")
        frames = load_video_frames(video_path, self.target_fps, self.image_size)

        # Load HDF5 trajectory
        hdf5_path = os.path.join(episode_dir, "trajectory.hdf5")
        traj = load_trajectory(hdf5_path)

        # Synchronize
        frames = sync_frames_to_trajectory(frames, len(traj['actions']))

        # Sample clip
        clip_frames, clip_actions, clip_states, clip_extrinsics = sample_clip(
            frames, traj['actions'], traj['states'], traj['camera_extrinsics'],
            self.frames_per_clip, self.stride
        )

        # Convert to tensors
        frames_tensor = torch.from_numpy(clip_frames).float() / 255.0  # [T, H, W, 3]
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)              # [T, 3, H, W]

        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_tensor = (frames_tensor - mean) / std

        return {
            'frames':     frames_tensor,                                          # [T, 3, H, W]
            'actions':    torch.from_numpy(clip_actions).float(),                 # [T, 7]
            'states':     torch.from_numpy(clip_states).float(),                  # [T, 7]
            'extrinsics': torch.from_numpy(clip_extrinsics).float(),              # [T, 4, 4]
        }
```

---

## Pose-to-Delta Conversion

DROID stores absolute end-effector poses in `states`. For V-JEPA 2-AC, convert to delta
actions (difference between consecutive poses):

```python
from scipy.spatial.transform import Rotation as R
import numpy as np

def pose_to_delta(states: np.ndarray) -> np.ndarray:
    """
    Convert absolute poses to delta actions.

    Args:
        states: [T, 7] -- [x, y, z, roll, pitch, yaw, gripper]

    Returns:
        deltas: [T-1, 7] -- delta actions between consecutive states
    """
    T = len(states)
    deltas = np.zeros((T - 1, 7), dtype=np.float32)

    for t in range(T - 1):
        # Position delta
        deltas[t, 0:3] = states[t+1, 0:3] - states[t, 0:3]

        # Orientation delta via rotation matrix composition
        r_t   = R.from_euler('xyz', states[t,   3:6])
        r_tp1 = R.from_euler('xyz', states[t+1, 3:6])
        r_delta = r_tp1 * r_t.inv()
        deltas[t, 3:6] = r_delta.as_euler('xyz')

        # Gripper delta
        deltas[t, 6] = states[t+1, 6] - states[t, 6]

    return deltas
```

---

## Camera Frame Transform

Transform world-frame poses into the camera coordinate frame for use as extrinsics conditioning:

```python
def world_to_camera_frame(world_pose: np.ndarray,
                           extrinsics: np.ndarray) -> np.ndarray:
    """
    Transform a 4x4 world-frame pose into camera frame.

    Args:
        world_pose:  [4, 4] -- homogeneous world pose
        extrinsics:  [4, 4] -- world-to-camera transform (DROID convention)

    Returns:
        camera_pose: [4, 4]
    """
    # DROID extrinsics: camera_T_world (transforms world points to camera)
    camera_pose = extrinsics @ world_pose
    return camera_pose

def extract_camera_6dof(extrinsics: np.ndarray) -> np.ndarray:
    """
    Extract 6-DOF (rotation 3 + translation 3) from 4x4 extrinsics for embedding.

    Args:
        extrinsics: [T, 4, 4]

    Returns:
        cam_6dof: [T, 6] -- [rx, ry, rz, tx, ty, tz]
    """
    T = len(extrinsics)
    cam_6dof = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        rot = R.from_matrix(extrinsics[t, :3, :3])
        cam_6dof[t, 0:3] = rot.as_euler('xyz')
        cam_6dof[t, 3:6] = extrinsics[t, :3, 3]
    return cam_6dof
```

---

## Validation Checklist

When loading a DROID episode, verify:

1. `frames.shape == (T, H, W, 3)` where T matches `len(actions)`
2. `actions.shape == (T, 7)` and `states.shape == (T, 7)`
3. `camera_extrinsics.shape == (T, 4, 4)`
4. All frames are finite (no NaN/Inf)
5. Gripper values in range [-1, 1]
6. Position values are physically plausible (within ~1m of workspace center)
