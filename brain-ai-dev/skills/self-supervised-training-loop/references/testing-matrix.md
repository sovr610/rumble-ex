# Testing Matrix: Self-Supervised Training Loop

## Overview

Seven testing phases covering all components of the self-supervised training loop. Total: 46 test scenarios (the implementation expands to 60+ with parametrize decorators).

---

## Phase 1: AMP Ordering (8 Tests)

Tests that verify the correct AMP operation sequence is enforced.

### 1.1 autocast_context_manager
- **Setup**: Create AMPContext with enabled=True, dtype=bfloat16
- **Action**: Call `ctx.autocast()` and check it returns a context manager
- **Assert**: Object is a context manager (has `__enter__` and `__exit__`)
- **Assert**: Inside context, tensors are cast to bfloat16

### 1.2 backward_scales_loss
- **Setup**: Create AMPContext, create simple tensor loss
- **Action**: Call `ctx.backward(loss)` — should call scaler.scale(loss).backward()
- **Assert**: No exception raised
- **Assert**: Gradient tensors exist on linear layer parameters after call

### 1.3 unscale_before_clip
- **Setup**: Create AMPContext, model, optimizer; run forward+backward
- **Action**: Call `ctx.unscale_and_clip(optimizer, model.parameters(), max_norm=1.0)`
- **Assert**: Gradients are at true scale (not multiplied by GradScaler's scale factor)
- **Assert**: Returned grad_norm is a finite float

### 1.4 clip_returns_correct_norm
- **Setup**: Inject known gradients into model parameters
- **Action**: Run `unscale_and_clip`, capture returned norm
- **Assert**: Returned value matches manually computed `sqrt(sum(g^2))`
- **Note**: norm returned is PRE-clip (before clipping is applied)

### 1.5 step_skips_on_inf
- **Setup**: Create AMPContext, model, optimizer; inject inf into a gradient
- **Action**: Run full sequence: backward → unscale_and_clip → step_and_update
- **Assert**: Parameters are unchanged after step (optimizer.step() was skipped)
- **Assert**: Scaler scale factor decreased (backoff applied)

### 1.6 update_adjusts_scale
- **Setup**: Run 2001 clean steps (no inf/nan) with AMPContext
- **Assert**: GradScaler scale factor has doubled at least once
- **Alternative**: Run one inf-trigger step, assert scale halved

### 1.7 zero_grad_set_to_none
- **Setup**: Create model, run forward+backward to populate .grad attributes
- **Action**: Call `optimizer.zero_grad(set_to_none=True)`
- **Assert**: `model.parameters()` all have `.grad is None` (not zero tensors)

### 1.8 gradient_accumulation
- **Setup**: Create AMPContext, 4 micro-batches
- **Action**: Run backward on each micro-batch without calling step/update; call step/update once at end
- **Assert**: Only one step is performed (not 4)
- **Assert**: No RuntimeError from double unscale_
- **Assert**: Final loss is finite

---

## Phase 2: EMA (8 Tests)

### 2.1 tau_at_step_0_equals_tau_base
- **Setup**: Create EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=100000)
- **Action**: Call `updater.get_tau(step=0)`
- **Assert**: result == 0.996 (exactly)

### 2.2 tau_at_total_steps_approaches_tau_final
- **Setup**: Same EMAUpdater
- **Action**: Call `updater.get_tau(step=100000)`
- **Assert**: result is close to 0.9999 (within 1e-6)

### 2.3 cosine_monotonicity
- **Setup**: Same EMAUpdater
- **Action**: Compute tau at steps 0, 10000, 20000, ..., 100000
- **Assert**: tau values are strictly monotonically increasing

### 2.4 no_grad_enforcement
- **Setup**: Create EMAUpdater, two small Linear models (online, target)
- **Action**: Call `updater.update(online, target, step=0)`
- **Assert**: No parameter in target has a `grad_fn` (no computation graph created)
- **Assert**: `torch.is_grad_enabled()` is True after call (no_grad was context, not global)

### 2.5 parameter_update_correctness
- **Setup**: online param = tensor([1.0]), target param = tensor([0.0]), tau=0.9
- **Action**: Apply EMA update manually and via EMAUpdater
- **Assert**: target param ≈ 0.9 * 0.0 + 0.1 * 1.0 = 0.1

### 2.6 buffer_update
- **Setup**: Create BatchNorm layers for online and target; set different running_mean values
- **Action**: Call `updater.update(online_bn, target_bn, step=0)`
- **Assert**: target running_mean equals online running_mean (buffers are COPIED, not EMA-averaged)

### 2.7 initial_sync
- **Setup**: Create EMAUpdater; create online (random init) and target (different random init)
- **Action**: Call `updater.initial_sync(online, target)`
- **Assert**: All target parameters equal corresponding online parameters (L2 distance = 0)

### 2.8 torch_lerp_equivalence
- **Setup**: Create two tensors a, b with known values; tau=0.996
- **Action**: Compute both `tau*a + (1-tau)*b` and `a.lerp_(b, 1-tau)` (on clones)
- **Assert**: Results are equal within numerical tolerance (1e-6)

---

## Phase 3: DDP Setup (6 Tests)

Note: These tests use mock/stub objects where actual process group setup is not available.

### 3.1 online_wrapped_has_module_attribute
- **Setup**: Create a simple model; call `wrap_online_model(model, rank=0, sync_batchnorm=False)` (mocked)
- **Assert**: Returned object has `.module` attribute
- **Assert**: `type(returned).__name__` is 'DistributedDataParallel' (mocked DDP)

### 3.2 target_not_wrapped
- **Setup**: Create target encoder
- **Assert**: Target encoder does NOT have `.module` attribute
- **Assert**: All parameters have `requires_grad=False`

### 3.3 syncbatchnorm_before_wrap
- **Setup**: Create model with nn.BatchNorm2d layers
- **Action**: Call `nn.SyncBatchNorm.convert_sync_batchnorm(model)` BEFORE DDP wrap
- **Assert**: All BatchNorm2d layers replaced with SyncBatchNorm2d layers
- **Assert**: No BatchNorm2d layers remain in the model

### 3.4 find_unused_parameters_false
- **Setup**: Mock DDP creation call; capture kwargs
- **Assert**: `find_unused_parameters=False` in the kwargs passed to DDP constructor

### 3.5 gradient_as_bucket_view_true
- **Setup**: Mock DDP creation call; capture kwargs
- **Assert**: `gradient_as_bucket_view=True` in the kwargs passed to DDP constructor

### 3.6 cleanup_destroys_process_group
- **Setup**: Mock `dist.destroy_process_group`; call `cleanup_distributed()`
- **Assert**: `destroy_process_group` was called exactly once
- **Assert**: No exception raised if called twice (safe cleanup)

---

## Phase 4: Optimizer/Schedule (6 Tests)

### 4.1 adamw_params_correct
- **Setup**: Create model; build optimizer via `build_optimizer(model, cfg)`
- **Assert**: `optimizer.param_groups[0]['lr']` == cfg.lr
- **Assert**: `optimizer.param_groups[0]['weight_decay']` == cfg.weight_decay
- **Assert**: `optimizer.param_groups[0]['betas']` == (0.9, 0.95)

### 4.2 warmup_linearity
- **Setup**: Create cosine warmup scheduler with warmup_steps=1000
- **Action**: Step scheduler from 0 to 999; record lr at each step
- **Assert**: lr values form a linear ramp from 0 to base_lr
- **Assert**: lr at step 500 ≈ 0.5 * base_lr

### 4.3 cosine_decay_shape
- **Setup**: Same scheduler; step from warmup_steps to total_steps; record lr
- **Assert**: lr monotonically decreases
- **Assert**: lr follows cosine shape (verify at step = warmup + half_decay_steps ≈ (base_lr + lr_min) / 2)

### 4.4 lr_min_at_end
- **Setup**: Step scheduler to total_steps
- **Assert**: Final lr == lr_min (within 1e-8)

### 4.5 scheduler_state_dict_round_trip
- **Setup**: Create scheduler; step 50 times; save state_dict
- **Action**: Create fresh scheduler; load state_dict; step 1 more time
- **Assert**: lr after resume == lr that would result from stepping the original 51 times

### 4.6 weight_decay_excludes_bias_norm
- **Setup**: Build optimizer with `build_optimizer(model, cfg)` that separates param groups
- **Assert**: Bias parameters have weight_decay=0.0
- **Assert**: LayerNorm/BatchNorm parameters have weight_decay=0.0
- **Assert**: Linear weight parameters have weight_decay=cfg.weight_decay

---

## Phase 5: Checkpointing (8 Tests)

### 5.1 all_six_states_saved
- **Setup**: Create all components; call `manager.save(model, target, optimizer, scaler, scheduler, step=100)`
- **Action**: Load the saved file with `torch.load()`
- **Assert**: File contains exactly the six keys: model_state_dict, target_state_dict, optimizer_state_dict, scaler_state_dict, scheduler_state_dict, step

### 5.2 auto_resume_finds_latest
- **Setup**: Create checkpoint files: checkpoint-100.pt, checkpoint-200.pt, checkpoint-300.pt
- **Action**: Call `manager.load_latest()`
- **Assert**: Returns checkpoint-300.pt (highest step)

### 5.3 ddp_unwrap_correct
- **Setup**: Create DDP-wrapped model
- **Action**: Call `manager.save(ddp_model, ...)`
- **Action**: Load checkpoint, check model_state_dict keys
- **Assert**: No keys start with 'module.' prefix (DDP was unwrapped before saving)

### 5.4 atomic_write
- **Setup**: Create checkpoint manager with a mock that raises OSError during rename
- **Action**: Simulate a crash during save
- **Assert**: No partial file left at the destination path
- **Assert**: Temp file is cleaned up

### 5.5 checkpoint_pruning
- **Setup**: Save 5 checkpoints (steps 100, 200, 300, 400, 500); keep_last=3
- **Action**: Call `manager.prune()`
- **Assert**: Only checkpoint-300.pt, checkpoint-400.pt, checkpoint-500.pt remain
- **Assert**: checkpoint-100.pt and checkpoint-200.pt are deleted

### 5.6 load_order_correct
- **Setup**: Save checkpoint at step 100; create fresh model+optimizer
- **Action**: Call `manager.resume(model, target, optimizer, scaler, scheduler)` in the correct order
- **Assert**: Returned step == 100
- **Assert**: Model weights match saved weights
- **Assert**: Optimizer lr matches saved lr

### 5.7 resume_step_continuity
- **Setup**: Train for 5 steps; save; resume; continue for 5 more steps
- **Assert**: Steps are numbered 0-4 before save, 5-9 after resume (continuous)
- **Assert**: Loss is finite throughout

### 5.8 corrupt_checkpoint_handling
- **Setup**: Write a checkpoint file with random bytes (corrupt)
- **Action**: Call `manager.load_latest()` on a directory containing only this file
- **Assert**: Raises a clear exception (not a cryptic internal error)
- **Assert**: Exception message indicates the checkpoint is unloadable

---

## Phase 6: W&B (6 Tests)

### 6.1 rank_nonzero_logs_nothing
- **Setup**: Create WandbLogger(cfg, rank=1)
- **Action**: Call log_step, log_predictions, finish
- **Assert**: No wandb module is imported (or wandb.init was never called)
- **Assert**: No exception raised

### 6.2 rank_zero_logs_all_metrics
- **Setup**: Create WandbLogger with mock wandb; call log_step with known values
- **Assert**: Mock wandb.log was called with the correct metric keys
- **Assert**: 'train/loss', 'train/grad_norm', 'train/ema_tau', 'train/lr', 'train/gpu_mem_gb' all present

### 6.3 prediction_grid_shape
- **Setup**: Create 16 images of shape (3, 32, 32); call log_predictions
- **Action**: Capture the grid tensor passed to wandb.Image
- **Assert**: Grid has shape (3, H, W) where H and W are multiples of image size
- **Assert**: Grid values are in [0, 1] range (normalized)

### 6.4 step_metric_defined
- **Setup**: Create WandbLogger with mock wandb
- **Assert**: `wandb.define_metric("train/step")` was called
- **Assert**: `wandb.define_metric("train/*", step_metric="train/step")` was called

### 6.5 alert_on_spike
- **Setup**: Create WandbLogger with mock wandb; simulate 10 steps with normal grad_norm=0.5
- **Action**: Log a step with grad_norm=50.0 (100x the average)
- **Assert**: `wandb.alert` was called with title indicating gradient spike

### 6.6 finish_safe_multiple_times
- **Setup**: Create WandbLogger with mock wandb
- **Action**: Call `logger.finish()` twice
- **Assert**: No exception raised on second call
- **Assert**: `wandb.finish()` called at most once (or is idempotent)

---

## Phase 7: Integration (4 Tests)

### 7.1 ten_step_finite_loss
- **Setup**: Create SelfSupervisedTrainer with small model (64-dim, 2 layers) and default config
- **Action**: Call `trainer.setup()`, then manually run 10 training steps
- **Assert**: All 10 loss values are finite (not nan, not inf)
- **Assert**: All 10 grad_norm values are positive and finite

### 7.2 checkpoint_save_load_resumes_correctly
- **Setup**: Train for 5 steps; save checkpoint at step 4; create new trainer; load checkpoint
- **Assert**: Resumed trainer starts from step 5
- **Assert**: Model weights in resumed trainer match saved weights
- **Assert**: Optimizer lr in resumed trainer matches saved lr

### 7.3 ema_diverges_from_online
- **Setup**: Train for 100 steps with EMAUpdater enabled
- **Action**: Compute L2 distance between online and target parameters
- **Assert**: Distance > 0.0 (target diverged from online during training)
- **Assert**: Distance is finite (no explosion)

### 7.4 multi_rank_smoke_test
- **Setup**: Mock DDP with world_size=2; simulate two ranks executing setup()
- **Assert**: Each rank creates its own model on its own device (mocked)
- **Assert**: Online model is DDP-wrapped for each rank
- **Assert**: Target model is NOT DDP-wrapped for any rank
- **Assert**: W&B init called exactly once (rank 0 only)

---

## Parametrize Decorators

The following parametrize decorators expand the above 46 scenarios to 60+ test cases:

```python
@pytest.mark.parametrize("amp_dtype", ["bfloat16", "float16"])
# Applies to all Phase 1 tests — tests both dtype paths

@pytest.mark.parametrize("tau_base,tau_final", [
    (0.996, 0.9999),
    (0.99, 0.999),
    (0.9, 0.999),
])
# Applies to Phase 2 tests — tests multiple tau configurations

@pytest.mark.parametrize("max_grad_norm", [0.5, 1.0, 5.0])
# Applies to clip tests — verifies clipping at different thresholds

@pytest.mark.parametrize("keep_last", [1, 3, 5])
# Applies to checkpoint pruning test

@pytest.mark.parametrize("total_steps", [1000, 10000, 100000])
# Applies to EMA schedule tests — verifies schedule at different scales
```
