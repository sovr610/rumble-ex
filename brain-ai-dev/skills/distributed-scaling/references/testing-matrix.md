# Testing Matrix — Reference for Distributed Scaling Skill

This document specifies the test scenarios, simulation strategies, and correctness criteria for validating the distributed training infrastructure of the `brain_ai` system. Use this as the canonical reference when writing, auditing, or debugging distributed tests.

---

## 1. Simulation Strategy

Multi-GPU testing without actual multi-GPU hardware is achieved through:

### 1.1 torch.multiprocessing.spawn

```python
import torch.multiprocessing as mp

def test_fn(rank, world_size, *args):
    """Function executed by each simulated rank."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        # Test logic here
        pass
    finally:
        dist.destroy_process_group()

def test_ddp_basic():
    world_size = 2
    mp.spawn(test_fn, args=(world_size,), nprocs=world_size, join=True)
```

Key details:
- Use `"gloo"` backend for CPU-only testing. NCCL requires GPUs.
- `mp.spawn` creates `nprocs` processes, each calling `test_fn` with its rank as the first argument.
- `join=True` blocks until all processes complete. Exceptions in any process are re-raised in the spawning process.

### 1.2 Single-Process Mocking

For unit tests that need to run quickly without spawning processes:

```python
class MockDistributed:
    """Mock torch.distributed for single-process testing."""

    def __init__(self):
        self._initialized = True
        self._rank = 0
        self._world_size = 1

    def is_initialized(self):
        return self._initialized

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def all_reduce(self, tensor, op=None):
        pass  # No-op for single process

    def barrier(self):
        pass

    def broadcast(self, tensor, src=0):
        pass
```

This allows testing the logic of DDPWrapper, GradientAccumulator, and other components without actually initializing process groups.

### 1.3 Simulated World Sizes

For testing correctness properties (gradient equivalence, sample coverage), simulate multiple ranks in a single process by running the sampler or accumulator with different rank values:

```python
def simulate_distributed_sampling(dataset_size, world_size):
    """Simulate what DistributedSampler produces for each rank."""
    dataset = list(range(dataset_size))
    all_indices = {}
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        all_indices[rank] = list(sampler)
    return all_indices
```

---

## 2. DDP Correctness Tests

### 2.1 Gradient Equivalence: DDP vs Single GPU

**Objective:** Verify that DDP produces the same gradients as single-GPU training with the full batch.

**Method:**
1. Create a small model (use `BrainAIConfig.minimal()`).
2. Fix seeds on all ranks.
3. Single GPU: forward pass with batch of size `micro_batch * world_size`, compute loss, backward.
4. DDP: each rank processes `micro_batch`, backward (DDP averages gradients).
5. Compare gradients element-wise (tolerance: 1e-5 for fp32, 1e-3 for fp16).

```python
def test_gradient_equivalence(rank, world_size):
    # Setup
    model = create_test_model()
    single_model = copy.deepcopy(model)

    # DDP
    ddp_model = DDP(model.to(rank), device_ids=[rank])
    data_local = full_data[rank::world_size]  # Partition data
    loss = criterion(ddp_model(data_local), target_local)
    loss.backward()

    # Single GPU (on rank 0 only)
    if rank == 0:
        loss_single = criterion(single_model(full_data), full_target)
        loss_single.backward()

        for (n1, p1), (n2, p2) in zip(
            ddp_model.module.named_parameters(),
            single_model.named_parameters()
        ):
            torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-5)
```

### 2.2 all_reduce Correctness

**Objective:** Verify that `all_reduce_metrics()` returns correct aggregated values.

**Tests:**
- Each rank contributes a different value; verify the mean is correct.
- Each rank contributes the same value; verify the result equals that value.
- Test with both SUM and MEAN operations.
- Test with scalar tensors and multi-element tensors.

### 2.3 find_unused_parameters

**Objective:** Verify that DDP handles unused parameters correctly.

**Tests:**
- Create a model with conditional paths (System 2 reasoning in BrainAI).
- Forward pass that skips one path.
- Verify backward completes without hanging.
- Verify unused parameter gradients are None (not zero).

### 2.4 SyncBatchNorm Verification

**Objective:** Verify that batch statistics are synchronized across ranks.

**Tests:**
- Each rank has different input data.
- After forward pass with SyncBatchNorm, the running mean and variance are identical across ranks.
- Compare against single-GPU batch statistics on the full batch.

---

## 3. FSDP Correctness Tests

### 3.1 Sharding Verification

**Objective:** Verify that parameters are correctly sharded across ranks.

**Tests:**
- Sum local parameter counts across ranks. Total should equal unsharded model size.
- Each rank's local parameter count should be approximately `total / world_size`.
- No rank should hold the full model (unless using NO_SHARD).

```python
def test_sharding_ratio(rank, world_size):
    model = create_test_model()
    total_unsharded = sum(p.numel() for p in model.parameters())

    fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
    local_count = sum(p.numel() for p in fsdp_model.parameters())

    # Gather counts
    counts = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(counts, torch.tensor([float(local_count)]))

    if rank == 0:
        total_sharded = sum(c.item() for c in counts)
        assert abs(total_sharded - total_unsharded) / total_unsharded < 0.01
        assert local_count < total_unsharded  # Sharding happened
```

### 3.2 Checkpoint Round-Trip

**Objective:** Verify that saving and loading a distributed checkpoint preserves model state.

**Tests:**
1. Train for 10 steps with FSDP.
2. Save sharded checkpoint with `torch.distributed.checkpoint`.
3. Create a fresh FSDP model.
4. Load the checkpoint.
5. Verify all parameters match (within precision tolerance).
6. Verify the loaded model produces the same output for the same input.

```python
def test_checkpoint_roundtrip(rank, world_size, tmp_dir):
    model = create_fsdp_model(rank)

    # Train a few steps
    for _ in range(10):
        train_step(model)

    # Save
    dcp.save({"model": model.state_dict()}, dcp.FileSystemWriter(tmp_dir))
    dist.barrier()

    # Load into fresh model
    fresh_model = create_fsdp_model(rank)
    dcp.load({"model": fresh_model.state_dict()}, dcp.FileSystemReader(tmp_dir))

    # Verify
    test_input = create_test_input(rank)
    out_original = model(test_input)
    out_loaded = fresh_model(test_input)
    torch.testing.assert_close(out_original, out_loaded)
```

### 3.3 Full State Dict Consolidation

**Objective:** Verify that a full state dict can be extracted from FSDP and loaded into a non-FSDP model.

**Tests:**
1. Create FSDP model and train.
2. Extract full state dict (offload to CPU, rank 0 only).
3. Create a plain (non-FSDP) model.
4. Load the full state dict.
5. Verify identical outputs.

### 3.4 Mixed Precision Forward/Backward

**Objective:** Verify that mixed precision does not cause numerical errors.

**Tests:**
- Forward pass in bf16/fp16 completes without NaN or Inf.
- Gradients are finite.
- Loss is in a reasonable range (not diverged).
- Compare bf16 output to fp32 output (within tolerance: atol=0.01 for bf16).

---

## 4. Gradient Accumulation Tests

### 4.1 Accumulation Equivalence

**Objective:** Verify that K accumulation steps with micro-batch M produce the same gradients as one step with batch K*M.

**Method:**
1. Create a model. Clone it.
2. Model A: forward with batch size K*M, backward once.
3. Model B: forward K times with batch size M, accumulating gradients (divide loss by K).
4. Compare gradients.

```python
def test_accumulation_equivalence():
    K, M = 4, 8  # 4 accumulation steps, micro-batch 8
    model_a = create_test_model()
    model_b = copy.deepcopy(model_a)

    # Model A: single large batch
    data_full = torch.randn(K * M, input_dim)
    target_full = torch.randint(0, num_classes, (K * M,))
    loss_a = criterion(model_a(data_full), target_full)
    loss_a.backward()

    # Model B: accumulated micro-batches
    for k in range(K):
        data_k = data_full[k*M:(k+1)*M]
        target_k = target_full[k*M:(k+1)*M]
        loss_k = criterion(model_b(data_k), target_k) / K
        loss_k.backward()

    # Compare
    for (n, pa), (_, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-5, rtol=1e-5)
```

### 4.2 Effective Batch Size Computation

**Tests:**
- `effective_batch_size(micro=4, world=8)` with `accum=16` equals `4 * 16 * 8 = 512`.
- Verify across all preset configurations (dev, 1B, 3B, 7B).

### 4.3 AMP Scaler Integration

**Tests:**
- GradScaler scale factor updates correctly after accumulation steps.
- When inf/nan detected: optimizer step is skipped, scale is reduced.
- Unscale is called exactly once per optimizer step, not per micro-step.

### 4.4 no_sync Correctness

**Tests (with mp.spawn):**
- During accumulation steps, verify no all_reduce communication occurs.
- On the final accumulation step, verify all_reduce happens.
- Compare results with and without no_sync (should be identical, but no_sync is faster).

---

## 5. Sampler Coverage Tests

### 5.1 No Duplicate Samples

**Objective:** Verify that across all ranks, no sample index appears more than once (excluding padding).

```python
def test_no_duplicates():
    for world_size in [1, 2, 4, 8]:
        dataset_size = 1000
        seen = set()
        duplicates = set()
        for rank in range(world_size):
            sampler = DistributedSampler(
                range(dataset_size), num_replicas=world_size, rank=rank,
                shuffle=False, drop_last=True,
            )
            for idx in sampler:
                if idx in seen:
                    duplicates.add(idx)
                seen.add(idx)
        assert len(duplicates) == 0, f"Duplicates at world_size={world_size}: {duplicates}"
```

### 5.2 Full Coverage

**Objective:** Verify that the union of all ranks' samples covers the entire dataset (or nearly so).

```python
def test_full_coverage():
    dataset_size = 1000
    world_size = 4
    all_indices = set()
    for rank in range(world_size):
        sampler = DistributedSampler(
            range(dataset_size), num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        all_indices.update(sampler)
    assert all_indices == set(range(dataset_size))
```

### 5.3 Epoch Shuffle Changes Order

```python
def test_shuffle_changes():
    dataset_size = 100
    world_size = 2
    sampler = DistributedSampler(
        range(dataset_size), num_replicas=world_size, rank=0, shuffle=True,
    )

    sampler.set_epoch(0)
    order_0 = list(sampler)

    sampler.set_epoch(1)
    order_1 = list(sampler)

    assert order_0 != order_1, "set_epoch() did not change shuffle order"
```

### 5.4 Balanced Partition

**Objective:** Verify that all ranks receive the same number of samples (within 1).

```python
def test_balanced_partition():
    for dataset_size in [100, 101, 1000, 1023]:
        for world_size in [1, 2, 3, 4, 7, 8]:
            counts = []
            for rank in range(world_size):
                sampler = DistributedSampler(
                    range(dataset_size), num_replicas=world_size, rank=rank,
                )
                counts.append(len(list(sampler)))
            assert max(counts) - min(counts) <= 1, (
                f"Unbalanced: {counts} for size={dataset_size}, world={world_size}"
            )
```

---

## 6. Integration Tests

### 6.1 End-to-End Training Loop

**Objective:** Verify that a full training loop (data loading, forward, backward, optimizer step) completes without errors.

**Test (with mp.spawn, gloo backend, CPU):**
1. Create `BrainAI` with `BrainAIConfig.minimal()`.
2. Wrap with DDP (gloo backend for CPU).
3. Create DistributedSampler and DataLoader with synthetic data.
4. Run 10 training steps.
5. Verify loss decreases.
6. Verify model parameters changed.

### 6.2 Checkpoint Save-Load-Resume

**Test:**
1. Train for 5 steps. Save checkpoint (rank 0).
2. Load checkpoint into fresh model.
3. Train for 5 more steps.
4. Verify total loss trajectory is consistent (no regression from load).

### 6.3 Multi-Phase Training

**Test:**
1. Run Phase 1 for 5 steps, save checkpoint.
2. Load into Phase 2 configuration, train 5 steps.
3. Verify encoder weights from Phase 1 are preserved.

---

## 7. Performance Tests

### 7.1 Scaling Efficiency

**Objective:** Measure throughput at different simulated world sizes.

**Method:** Since we simulate on CPU, measure relative overhead rather than absolute throughput. Run the same computation at world_size 1 and 2 (using gloo). The ratio should be near 2x for compute-bound workloads.

### 7.2 Communication Overhead

**Objective:** Measure the time spent in all_reduce vs compute.

**Method:** Time the backward pass with and without `no_sync()`. The difference is the communication time.

### 7.3 Memory Profiling

**Objective:** Verify that FSDP reduces per-GPU memory as expected.

**Method:** Compare `torch.cuda.memory_allocated()` (or mock equivalent) between DDP and FSDP configurations.

---

## 8. Test Matrix Summary

| Category | Test Count | Requires spawn | Requires GPU | Priority |
|----------|-----------|---------------|-------------|----------|
| DDP gradient equivalence | 5 | Yes (gloo) | No | P0 |
| DDP all_reduce metrics | 4 | Yes (gloo) | No | P0 |
| DDP find_unused_parameters | 3 | Yes (gloo) | No | P1 |
| FSDP sharding verification | 4 | Yes (gloo) | No | P0 |
| FSDP checkpoint round-trip | 3 | Yes (gloo) | No | P0 |
| FSDP full state dict | 3 | Yes (gloo) | No | P1 |
| FSDP mixed precision | 4 | Yes (gloo) | No | P1 |
| Gradient accumulation equiv | 5 | No | No | P0 |
| AMP scaler integration | 4 | No | No | P1 |
| Sampler no-duplicates | 5 | No | No | P0 |
| Sampler full coverage | 4 | No | No | P0 |
| Sampler set_epoch | 3 | No | No | P0 |
| End-to-end training | 3 | Yes (gloo) | No | P0 |
| Checkpoint save-load | 3 | Optional | No | P1 |
| Config validation | 10 | No | No | P0 |
| Config presets | 5 | No | No | P1 |
| **Total** | **~68+** | | | |

The gen_distributed_tests.py script generates 100+ tests from this matrix by parametrizing over world sizes, model scales, sharding strategies, and accumulation step counts.
