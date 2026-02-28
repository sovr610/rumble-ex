# Testing Matrix: Distributed Memory Scaling Skill

## Overview

This document defines the complete test matrix for the distributed-memory-scaling skill. Tests are organized into six phases, covering config validation, strategy routing, FSDP wrapping, DeepSpeed wrapping, checkpoint portability, and scaling benchmarks. Each phase includes specific assertions, edge cases, and the expected pass/fail criteria.

---

## Phase 1: Config Validation

Tests for `DistributedConfig`, `FSDPConfig`, `DeepSpeedConfig`, and `OptimizerConfig` dataclasses.

### 1.1 Strategy Values

| Test | Input | Expected |
|------|-------|----------|
| Valid strategy: ddp | `strategy="ddp"` | No exception |
| Valid strategy: fsdp | `strategy="fsdp"` | No exception |
| Valid strategy: deepspeed_zero2 | `strategy="deepspeed_zero2"` | No exception |
| Valid strategy: deepspeed_zero3 | `strategy="deepspeed_zero3"` | No exception |
| Invalid strategy: unknown | `strategy="unknown_strategy"` | `ValueError` |
| Invalid strategy: empty string | `strategy=""` | `ValueError` |
| Invalid strategy: None | `strategy=None` | `TypeError` |

### 1.2 FSDP Sharding Strategies

| Test | Input | Expected |
|------|-------|----------|
| Valid: FULL_SHARD | `sharding_strategy="FULL_SHARD"` | No exception |
| Valid: SHARD_GRAD_OP | `sharding_strategy="SHARD_GRAD_OP"` | No exception |
| Valid: NO_SHARD | `sharding_strategy="NO_SHARD"` | No exception |
| Valid: HYBRID_SHARD | `sharding_strategy="HYBRID_SHARD"` | No exception |
| Invalid: typo | `sharding_strategy="FULL_SHARED"` | `ValueError` |

### 1.3 FSDP Wrap Policy Validation

| Test | Input | Expected |
|------|-------|----------|
| transformer_block without classes | `wrap_policy="transformer_block", wrap_module_classes=[]` | `ValueError` |
| transformer_block with classes | `wrap_policy="transformer_block", wrap_module_classes=["TransformerBlock"]` | No exception |
| size_based without classes | `wrap_policy="size_based", wrap_module_classes=[]` | No exception |
| size_based with classes (classes ignored) | `wrap_policy="size_based", wrap_module_classes=["X"]` | No exception (warning) |
| Invalid policy name | `wrap_policy="invalid"` | `ValueError` |

### 1.4 DeepSpeed Stage Validation

| Test | Input | Expected |
|------|-------|----------|
| Valid stage 2 | `zero_stage=2` | No exception |
| Valid stage 3 | `zero_stage=3` | No exception |
| Invalid stage 1 | `zero_stage=1` | `ValueError` |
| Invalid stage 4 | `zero_stage=4` | `ValueError` |

### 1.5 Cross-Field Validation

| Test | Input | Expected |
|------|-------|----------|
| offload_param on stage 3 | `zero_stage=3, offload_param="cpu"` | No exception |
| offload_param on stage 2 | `zero_stage=2, offload_param="cpu"` | `ValueError` |
| offload_optimizer on stage 2 | `zero_stage=2, offload_optimizer="cpu"` | No exception |
| offload_optimizer on stage 3 | `zero_stage=3, offload_optimizer="cpu"` | No exception |
| nvme offload without nvme path | `offload_param="nvme", nvme_path=None` | `ValueError` |

### 1.6 Serialization Round-Trip

| Test | Input | Expected |
|------|-------|----------|
| DistributedConfig to_dict/from_dict | Any valid config | Reconstructed config equals original |
| FSDPConfig to_dict/from_dict | Any valid FSDP config | Reconstructed equals original |
| DeepSpeedConfig to_dict/from_dict | Stage 2 and Stage 3 configs | Reconstructed equals original |
| OptimizerConfig to_dict/from_dict | Any optimizer config | Reconstructed equals original |

---

## Phase 2: Strategy Router

Tests for `StrategyRouter` class including wrapping order enforcement and path dispatch.

### 2.1 Wrapping Order Enforcement

| Test | Scenario | Expected |
|------|----------|----------|
| wrap_model before setup_distributed | Call wrap_model without setup_distributed | `RuntimeError: setup_distributed must be called first` |
| build_optimizer before wrap_model | Call build_optimizer without wrap_model | `RuntimeError: wrap_model must be called before build_optimizer` |
| load_checkpoint before wrap_model | Call load_checkpoint before wrap_model | `RuntimeError: wrap_model must be called before load_checkpoint` |
| Correct order completes | setup → wrap → build_optimizer → load | No exception |

### 2.2 DDP Path

| Test | Config | Expected |
|------|--------|----------|
| DDP dispatch | `strategy="ddp"` | Returns `DistributedDataParallel` wrapper |
| DDP context | `strategy="ddp"` | `StrategyContext.strategy == "ddp"` |
| DDP optimizer over wrapped params | `strategy="ddp"` | Optimizer param groups reference DDP module params |
| DDP save checkpoint | `strategy="ddp"` | Saves model state dict directly |

### 2.3 FSDP Path

| Test | Config | Expected |
|------|--------|----------|
| FSDP dispatch | `strategy="fsdp"` | Calls `FSDPWrapper.wrap()` |
| FSDP context strategy | `strategy="fsdp"` | `StrategyContext.strategy == "fsdp"` |
| FSDP optimizer created after wrap | `strategy="fsdp"` | Optimizer references FSDP module params |
| FSDP export_portable_weights | `strategy="fsdp"` | Calls `save_full_state_dict` |

### 2.4 DeepSpeed Path

| Test | Config | Expected |
|------|--------|----------|
| ZeRO-2 dispatch | `strategy="deepspeed_zero2"` | Calls `DeepSpeedWrapper.wrap()` with `zero_stage=2` |
| ZeRO-3 dispatch | `strategy="deepspeed_zero3"` | Calls `DeepSpeedWrapper.wrap()` with `zero_stage=3` |
| DeepSpeed context strategy | `strategy="deepspeed_zero3"` | `StrategyContext.strategy == "deepspeed_zero3"` |
| DeepSpeed export_portable_weights | `strategy="deepspeed_zero3"` | Calls `export_fp32_weights` |

---

## Phase 3: FSDP Wrapper

Tests for `FSDPWrapper` class.

### 3.1 Wrap Policy Selection

| Test | Config | Expected |
|------|--------|----------|
| transformer_block policy | `wrap_policy="transformer_block", wrap_module_classes=["TransformerBlock"]` | `ModuleWrapPolicy` instantiated with `TransformerBlock` class |
| size_based policy | `wrap_policy="size_based"` | `size_based_auto_wrap_policy` used |
| Verify FSDP units | transformer_block policy on 4-block model | `print(model)` shows 4 individual FSDP units |

### 3.2 Mixed Precision Config

| Test | Config | Expected |
|------|--------|----------|
| bf16 precision | `mixed_precision="bf16"` | `MixedPrecision(param_dtype=bfloat16, reduce_dtype=bfloat16, buffer_dtype=bfloat16)` |
| fp16 precision | `mixed_precision="fp16"` | `MixedPrecision(param_dtype=float16, reduce_dtype=float16, buffer_dtype=float16)` |
| none precision | `mixed_precision="none"` | `mixed_precision=None` passed to FSDP |
| Invalid precision | `mixed_precision="fp8"` | `ValueError` |

### 3.3 Activation Checkpointing

| Test | Config | Expected |
|------|--------|----------|
| Checkpointing off | `activation_checkpointing="off"` | `apply_activation_checkpointing` not called |
| Checkpointing on | `activation_checkpointing="transformer_block"` | `apply_activation_checkpointing` called AFTER FSDP wrap |
| Checkpointing before wrap error | Call `apply_activation_checkpointing` before wrap | Wrap order assertion fails |

### 3.4 Full State Dict

| Test | Scenario | Expected |
|------|----------|----------|
| save_full_state_dict config | Call `save_full_state_dict` | Uses `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` |
| save_full_state_dict context type | Call `save_full_state_dict` | Uses `StateDictType.FULL_STATE_DICT` |
| load_full_state_dict | Load then compare params | Loaded params match saved params (on rank 0) |

### 3.5 Sharded Checkpoint

| Test | Scenario | Expected |
|------|----------|----------|
| save_sharded_checkpoint | Save then list directory | Directory contains per-rank shard files |
| sharded checkpoint round-trip | Save and load | Model params match after load |
| DCP writer called | Save sharded | `dist_cp.FileSystemWriter` instantiated with correct path |

---

## Phase 4: DeepSpeed Wrapper

Tests for `DeepSpeedWrapper` class.

### 4.1 JSON Generation for Stage 2

| Test | Config | Expected JSON fields |
|------|--------|---------------------|
| Stage 2 base | `zero_stage=2` | `zero_optimization.stage == 2` |
| Stage 2 bucket | `reduce_bucket_size=1e9` | `zero_optimization.reduce_bucket_size == 1000000000` |
| Stage 2 comm flags | default | `overlap_comm=true, contiguous_gradients=true, reduce_scatter=true` |
| Stage 2 no stage3 fields | `zero_stage=2` | `stage3_prefetch_bucket_size` NOT present |
| Stage 2 no offload_param | `zero_stage=2` | `offload_param` NOT present |

### 4.2 JSON Generation for Stage 3

| Test | Config | Expected |
|------|--------|----------|
| Stage 3 fields present | `zero_stage=3` | `stage3_prefetch_bucket_size`, `stage3_param_persistence_threshold` present |
| Stage 3 offload_optimizer CPU | `zero_stage=3, offload_optimizer="cpu"` | `offload_optimizer.device == "cpu"` |
| Stage 3 offload_param CPU | `zero_stage=3, offload_param="cpu"` | `offload_param.device == "cpu"` |
| Stage 3 offload_param NVMe | `zero_stage=3, offload_param="nvme"` | `offload_param.device == "nvme"` |

### 4.3 Stage-Specific Field Validation

| Test | Config | Expected |
|------|--------|----------|
| offload_param stage 2 rejected | `zero_stage=2, offload_param="cpu"` | `ValueError` before JSON generation |
| Bucket sizes positive | `reduce_bucket_size=-1` | `ValueError` |

### 4.4 Engine Mock

| Test | Scenario | Expected |
|------|----------|----------|
| wrap() calls deepspeed.initialize | Mock deepspeed | `deepspeed.initialize` called with correct config dict |
| save_checkpoint calls engine | Mock engine | `engine.save_checkpoint` called with correct save_dir and tag |
| load_checkpoint calls engine | Mock engine | `engine.load_checkpoint` called with correct load_dir and tag |

---

## Phase 5: Checkpoint Portability

Tests for the end-to-end checkpoint conversion and resume paths.

### 5.1 FSDP Full State Dict Round-Trip

| Test | Scenario | Expected |
|------|----------|----------|
| Save and load full state dict | Save → clean model → load | All parameter values match within float32 tolerance |
| Parameter names match | After load | State dict keys identical to pre-save keys |
| Rank 0 only file exists | After save | Only 1 checkpoint file created (not one per rank) |

### 5.2 FSDP Sharded Checkpoint Round-Trip

| Test | Scenario | Expected |
|------|----------|----------|
| Save and load sharded | Save → clean state → load | All parameter values match |
| Step counter preserved | Include step in state dict | Loaded step matches saved step |
| Optimizer state preserved | Include optimizer in state dict | Optimizer state matches after load |

### 5.3 zero_to_fp32 Mock

| Test | Scenario | Expected |
|------|----------|----------|
| export_fp32_weights called | Mock subprocess | `zero_to_fp32.py` invoked with correct checkpoint_dir and output_path |
| fp32 output loadable | Mock output file | `torch.load` succeeds and returns state dict |

### 5.4 Cross-Strategy Load

| Test | Scenario | Expected |
|------|----------|----------|
| FSDP full → unwrapped load | FSDP save → load into plain nn.Module | `model.load_state_dict` succeeds, no missing/unexpected keys |
| FSDP full → DDP load | FSDP save → DDP wrap | DDP-wrapped model has correct parameters |
| DS fp32 → FSDP load | zero_to_fp32 output → sync_module_states wrap | FSDP model has correct parameters |

---

## Phase 6: Scaling Benchmark

Tests for `ScalingBenchmark` class and metrics output.

### 6.1 Single-GPU Measurement Mock

| Test | Scenario | Expected |
|------|----------|----------|
| run_single returns BenchResult | Mock training loop | `BenchResult.throughput > 0` |
| run_single step count | 50 measured steps | `BenchResult.measured_steps == 50` |
| run_single warmup excluded | 10 warmup steps | First 10 steps not in statistics |

### 6.2 Multi-GPU Measurement Mock

| Test | Scenario | Expected |
|------|----------|----------|
| run_multi returns BenchResult | Mock with world_size=4 | `BenchResult.world_size == 4` |
| run_multi throughput scales | 4 GPUs, perfect scaling | `throughput_n ≈ 4 * throughput_1` |

### 6.3 Efficiency Computation

| Test | Input | Expected |
|------|-------|----------|
| Perfect scaling | `throughput_1=1000, throughput_n=4000, N=4` | `scaling_efficiency == 1.0` |
| 90% scaling | `throughput_1=1000, throughput_n=3600, N=4` | `scaling_efficiency == 0.9` |
| Superlinear | `throughput_1=1000, throughput_n=4200, N=4` | `scaling_efficiency == 1.05` |
| Zero throughput guard | `throughput_1=0` | `ZeroDivisionError` or handled with `ValueError` |
| world_size=1 efficiency | `throughput_1=1000, throughput_n=1000, N=1` | `scaling_efficiency == 1.0` |

### 6.4 Metrics Output Schema

| Test | Scenario | Expected |
|------|----------|----------|
| Required fields present | `save_metrics()` | All 6 required fields in output JSON |
| Strategy field type | Any strategy | `strategy` is a string |
| world_size field type | Any world_size | `world_size` is an int |
| scaling_efficiency range | Valid run | `0.0 < scaling_efficiency <= 2.0` |
| memory_peak_gb positive | Valid run | `memory_peak_gb > 0` |
| throughput_p50 positive | Valid run | `throughput_p50 > 0` |
| step_time_p50_ms positive | Valid run | `step_time_p50_ms > 0` |
| JSON parseable | `save_metrics()` | Output file parseable by `json.loads` |

---

## Edge Cases

### Mismatched world_size

| Test | Scenario | Expected |
|------|----------|----------|
| Sharded checkpoint wrong world_size | Save 8-GPU, load 4-GPU | Descriptive error, not silent corruption |
| DS checkpoint wrong world_size | Save 8-GPU, load 4-GPU | `engine.load_checkpoint` raises or warns |

### Missing Process Group

| Test | Scenario | Expected |
|------|----------|----------|
| wrap_model without init_process_group | Call wrap_model before dist.init | `RuntimeError: Default process group not initialized` |
| save_full_state_dict without process group | Call on unwrapped model | Descriptive error message |

### Invalid Wrap Classes

| Test | Scenario | Expected |
|------|----------|----------|
| wrap_module_classes contains non-existent class | `wrap_module_classes=["NonExistentClass"]` | `ValueError` at config validation time, not silently at wrap time |
| wrap_module_classes contains string not importable | `wrap_module_classes=["my.module.Foo"]` | `ImportError` or `ValueError` with message |

### Offload on Non-Stage-3

| Test | Scenario | Expected |
|------|----------|----------|
| offload_param=cpu on ZeRO-2 | Validate config | `ValueError` with message: `offload_param requires zero_stage=3` |
| offload_param=nvme on ZeRO-2 | Validate config | Same `ValueError` |
| offload_optimizer=cpu on ZeRO-2 | Validate config | No exception (optimizer offload is valid for stage 2) |

### Benchmark Edge Cases

| Test | Scenario | Expected |
|------|----------|----------|
| world_size=1 scaling | Run single then compute efficiency | `scaling_efficiency == 1.0` (not undefined) |
| All steps are warmup | `warmup_steps >= measured_steps` | `ValueError: no steps to measure` |
| Memory measurement unavailable | Non-CUDA device | `memory_peak_gb = 0.0` or `None` with warning |
