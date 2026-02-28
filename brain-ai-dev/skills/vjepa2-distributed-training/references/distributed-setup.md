# Distributed Setup Reference

## Overview

V-JEPA 2 distributed training supports two launch modes:
1. **SLURM cluster** — auto-detected via environment variables set by the SLURM scheduler
2. **Local multi-GPU** — launched via `torchrun` on a single machine

Both modes use NCCL as the backend for GPU-to-GPU communication.

---

## SLURM Auto-Detection

SLURM sets specific environment variables for each task in a job array. Detection logic checks for the presence of these variables:

```python
SLURM_NTASKS      # Total number of tasks (== world_size)
SLURM_PROCID      # Global rank of this task (0-indexed)
SLURM_LOCALID     # Local rank within the node (0-indexed, used to set CUDA device)
SLURM_NODEID      # Node index within the job
SLURM_JOB_ID      # Job ID (used to construct rendezvous tmpdir path)
```

Detection pattern:

```python
def is_slurm() -> bool:
    return (
        "SLURM_NTASKS" in os.environ
        and "SLURM_PROCID" in os.environ
        and "SLURM_LOCALID" in os.environ
    )
```

---

## NCCL Process Group Initialization

### Backend

Always use `"nccl"` backend for GPU training. CPU-only fallback uses `"gloo"`.

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")
```

### Init Methods

Three supported init methods in priority order:

1. **env://** (default for torchrun and most SLURM setups)
   - Requires `MASTER_ADDR` and `MASTER_PORT` in environment
   - `torchrun` sets these automatically

2. **file://** (SLURM tmpdir rendezvous)
   - Use when `SLURM_JOB_TMPDIR` is set (local NVMe scratch on compute nodes)
   - Avoids NFS contention for the rendezvous file
   - Path: `file://${SLURM_JOB_TMPDIR}/dist_init_${SLURM_JOB_ID}`

3. **Single-process fallback**
   - When no distributed env vars present
   - Skip `init_process_group` entirely
   - rank=0, local_rank=0, world_size=1

### SLURM Init Code Pattern

```python
import os
import torch
import torch.distributed as dist

def init_distributed_slurm():
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # Use SLURM_JOB_TMPDIR if available for rendezvous (local NVMe, avoids NFS)
    job_tmpdir = os.environ.get("SLURM_JOB_TMPDIR", "/tmp")
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    init_file = os.path.join(job_tmpdir, f"dist_init_{job_id}")
    init_url = f"file://{init_file}"

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=init_url,
        rank=rank,
        world_size=world_size,
    )

    return rank, local_rank, world_size
```

### Local Multi-GPU (torchrun) Init Code Pattern

`torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` automatically.

```python
def init_distributed_local():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return rank, local_rank, world_size
```

---

## Fallback: Single-Process Mode

When neither SLURM env vars nor torchrun env vars are present, the system falls back to single-process mode. This enables running on a single GPU or CPU without modification to the training code.

```python
def init_single_process():
    # No dist.init_process_group call
    # CUDA device defaults to 0 if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 0, 1  # rank, local_rank, world_size
```

The caller checks `dist.is_initialized()` before calling any distributed ops.

---

## Cleanup

Always call cleanup at the end of training or on exception to release NCCL resources:

```python
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
```

Cleanup is idempotent — safe to call multiple times.

---

## torchrun Launch Commands

### Single node, 8 GPUs

```bash
torchrun \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py --config config.yaml
```

### Multi-node (2 nodes, 8 GPUs each)

```bash
# On node 0 (master):
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<node0_ip> \
    --master_port=29500 \
    train.py --config config.yaml

# On node 1:
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<node0_ip> \
    --master_port=29500 \
    train.py --config config.yaml
```

---

## Process Group Utilities

```python
def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()
```

---

## Environment Variable Reference

| Variable | Source | Description |
|----------|--------|-------------|
| `SLURM_NTASKS` | SLURM | Total tasks = world_size |
| `SLURM_PROCID` | SLURM | Global rank |
| `SLURM_LOCALID` | SLURM | Local rank on node |
| `SLURM_NODEID` | SLURM | Node index |
| `SLURM_JOB_ID` | SLURM | Job identifier |
| `SLURM_JOB_TMPDIR` | SLURM | Node-local tmpdir (NVMe) |
| `RANK` | torchrun | Global rank |
| `LOCAL_RANK` | torchrun | Local rank on node |
| `WORLD_SIZE` | torchrun | Total processes |
| `MASTER_ADDR` | torchrun | Rendezvous hostname |
| `MASTER_PORT` | torchrun | Rendezvous port |

---

## Common Pitfalls

1. **Hanging at init**: All ranks must call `init_process_group` with the same arguments. A rank mismatch or missing rank causes indefinite blocking.

2. **CUDA device assignment**: Set `torch.cuda.set_device(local_rank)` BEFORE creating any tensors or calling `init_process_group`. Otherwise NCCL may use the wrong GPU.

3. **NFS rendezvous contention**: On large clusters, NFS rendezvous files can cause slow initialization. Use `SLURM_JOB_TMPDIR` (local NVMe) when available.

4. **Port conflicts**: Multiple jobs on the same node may collide on `MASTER_PORT`. Use different ports per job or rely on SLURM's tmpdir init method.

5. **Cleanup on exception**: Wrap training in try/finally to ensure `dist.destroy_process_group()` is called even on crashes.
