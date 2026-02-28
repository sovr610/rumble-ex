# SLURM + submitit Integration Reference

## Overview

[submitit](https://github.com/facebookincubator/submitit) is a Python library for submitting jobs to SLURM clusters from Python code. It handles job submission, output logging, and — critically for long training runs — **preemption-safe checkpointing via auto-requeue**.

V-JEPA 2 uses submitit to submit distributed training jobs that can be safely preempted (e.g., by higher-priority jobs) and automatically requeued to resume from the last checkpoint.

---

## Core Concepts

### The Trainer Callable

The job's entrypoint must be a **callable class** (not a plain function) because submitit needs to serialize and deserialize it across resubmissions. The class must implement:

- `__call__(self)` — runs the actual training
- `checkpoint(self)` — called by submitit on preemption signal (SIGTERM); returns a `DelayedSubmission` that requeues the job

```python
import submitit

class Trainer:
    def __init__(self, args: dict):
        self.args = args

    def __call__(self) -> None:
        """Entrypoint for the actual training logic."""
        setup_distributed()
        model = build_model(self.args)
        # Resume from checkpoint if available
        ckpt_path = self.args.get("resume_checkpoint")
        if ckpt_path and os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path)
        train(model, self.args)

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        """Called when SLURM sends SIGTERM (preemption warning).
        Saves current state and returns a DelayedSubmission to requeue.
        """
        # The job will be resubmitted with the same Trainer instance
        # (current args already contain the checkpoint path)
        return submitit.helpers.DelayedSubmission(self)
```

### DelayedSubmission for Auto-Requeue

When SLURM preempts a job (e.g., walltime exceeded or higher-priority job), it sends SIGTERM. submitit intercepts this signal, calls `trainer.checkpoint()`, and uses the returned `DelayedSubmission` to immediately resubmit the job with the same parameters.

The resubmitted job picks up from the last checkpoint because `self.args` contains the checkpoint path (updated during training).

---

## SLURMSubmitter

### Configuration

```python
from dataclasses import dataclass

@dataclass
class SLURMConfig:
    nodes: int = 1                    # Number of nodes
    gpus_per_node: int = 8            # GPUs per node
    mem_per_gpu: str = "64G"          # Memory per GPU
    timeout_min: int = 4320           # 72 hours in minutes
    partition: str = "learn"          # SLURM partition
    account: str = ""                 # SLURM account (billing)
    qos: str = ""                     # Quality of service
    cpus_per_task: int = 10           # CPU cores per task
    slurm_signal_delay_s: int = 120   # Seconds before timeout to send SIGTERM
```

### Executor Setup

```python
import submitit

def create_executor(config: SLURMConfig, log_dir: str) -> submitit.AutoExecutor:
    executor = submitit.AutoExecutor(folder=log_dir)

    slurm_kwargs = {
        "slurm_gres": f"gpu:{config.gpus_per_node}",
        "slurm_cpus_per_task": config.cpus_per_task,
        "slurm_mem_per_gpu": config.mem_per_gpu,
        "slurm_signal_delay_s": config.slurm_signal_delay_s,
    }

    if config.account:
        slurm_kwargs["slurm_account"] = config.account
    if config.qos:
        slurm_kwargs["slurm_qos"] = config.qos

    executor.update_parameters(
        nodes=config.nodes,
        gpus_per_node=config.gpus_per_node,
        tasks_per_node=config.gpus_per_node,  # one task per GPU
        timeout_min=config.timeout_min,
        slurm_partition=config.partition,
        **slurm_kwargs,
    )

    return executor
```

### Submitting a Job

```python
def submit(trainer: Trainer, executor: submitit.AutoExecutor) -> str:
    job = executor.submit(trainer)
    print(f"Submitted job: {job.job_id}")
    return job.job_id
```

---

## Code Snapshotting

Code snapshotting copies the entire source tree to the experiment directory before launching. This ensures:

1. **Reproducibility** — the exact code used to produce a checkpoint is preserved alongside it
2. **Requeue safety** — if the source tree changes between submission and requeue, the job still uses the original code
3. **Debugging** — postmortem analysis has the exact code version

### Implementation

```python
import shutil
import os
import glob

def snapshot_code(src_dir: str, dest_dir: str, extensions: list = None) -> None:
    """Copy source tree to dest_dir, preserving directory structure."""
    if extensions is None:
        extensions = [".py", ".yaml", ".yml", ".json", ".sh"]

    os.makedirs(dest_dir, exist_ok=True)

    for ext in extensions:
        for src_path in glob.glob(os.path.join(src_dir, "**", f"*{ext}"), recursive=True):
            # Skip hidden directories and __pycache__
            if any(part.startswith(".") or part == "__pycache__"
                   for part in src_path.split(os.sep)):
                continue

            rel_path = os.path.relpath(src_path, src_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)

    print(f"Code snapshotted to: {dest_dir}")
```

### Usage Pattern

```python
experiment_dir = f"/checkpoint/{user}/vjepa2/{experiment_name}"
code_snapshot_dir = os.path.join(experiment_dir, "code_snapshot")

snapshot_code(
    src_dir=os.path.dirname(os.path.abspath(__file__)),
    dest_dir=code_snapshot_dir,
)
```

---

## Full Submit Workflow

```python
def run_experiment(config: dict, slurm_config: SLURMConfig):
    # 1. Create experiment directory
    experiment_dir = f"/checkpoint/{os.getenv('USER')}/vjepa2/{config['experiment_name']}"
    log_dir = os.path.join(experiment_dir, "submitit_logs")
    os.makedirs(log_dir, exist_ok=True)

    # 2. Snapshot code for reproducibility
    code_dir = os.path.join(experiment_dir, "code_snapshot")
    snapshot_code(src_dir=".", dest_dir=code_dir)

    # 3. Update config with checkpoint path
    config["checkpoint_dir"] = experiment_dir

    # 4. Create trainer callable
    trainer = Trainer(args=config)

    # 5. Create executor and submit
    executor = create_executor(slurm_config, log_dir)
    job_id = submit(trainer, executor)

    print(f"Experiment: {experiment_dir}")
    print(f"SLURM Job ID: {job_id}")
    print(f"Logs: {log_dir}/{job_id}_*.out")
    return job_id
```

---

## AutoExecutor vs SlurmExecutor

submitit provides two executor classes:

- **`AutoExecutor`** — automatically selects between SLURM and local execution based on environment. Use in code that runs both on clusters and developer machines.
- **`SlurmExecutor`** — SLURM-only. Use when you always submit to SLURM.

For V-JEPA 2, `AutoExecutor` is preferred because the same code can be tested locally.

---

## Monitoring Jobs

```python
# After submission:
job = executor.submit(trainer)

# Check status
print(job.state)       # RUNNING, PENDING, DONE, FAILED

# Wait for completion (blocking)
result = job.result()

# Get log paths
print(job.paths.stdout)
print(job.paths.stderr)
```

---

## SLURM Script Equivalent

For reference, the equivalent raw SLURM script for a 2-node, 8-GPU/node job:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=64G
#SBATCH --time=72:00:00
#SBATCH --partition=learn
#SBATCH --output=/checkpoint/%u/vjepa2/%x/%j_%t.out
#SBATCH --signal=USR1@120  # Send signal 120 seconds before timeout

srun python train.py --config config.yaml --checkpoint-dir /checkpoint/$USER/vjepa2/$SLURM_JOB_NAME
```

submitit generates and manages this script automatically.

---

## Common Issues

1. **Pickling errors**: The `Trainer` class and all its attributes must be picklable (no lambda functions, open file handles, or un-picklable objects as attributes).

2. **Timeout too short**: If training doesn't checkpoint frequently enough, a preemption can lose significant work. Checkpoint at least every 30 minutes.

3. **Log directory permissions**: The log directory must be writable by the submitting user and accessible from compute nodes (shared filesystem required).

4. **Missing `slurm_signal_delay_s`**: Without this, SLURM won't send SIGTERM before killing the job, so `checkpoint()` never gets called. Always set to at least 60 seconds.

5. **Stale rendezvous files**: If a job crashes without cleanup, the rendezvous file in `SLURM_JOB_TMPDIR` may persist. Use unique paths per job ID to avoid conflicts.
