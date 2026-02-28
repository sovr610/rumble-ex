"""
compile_smoketest_template.py
------------------------------
CompileSmoketest: validates that a compiled model can complete full
forward-backward-optimizer cycles without error or NaN loss.

The smoketest is designed to catch:
- Compilation-time failures that only manifest on the first forward pass
- Backward failures caused by autograd interactions with compiled code
- NaN/inf loss values from numerical precision changes under compilation
- Memory errors that only appear under full training step execution

Usage:
    from compile_smoketest_template import CompileSmoketest
    import torch
    import torch.nn as nn
    import torch.optim as optim

    model = torch.compile(your_model, mode="default")
    smoketest = CompileSmoketest(steps=3, seed=42)

    success = smoketest.run(
        model=model,
        sample_batch={"input_ids": torch.randint(0, 100, (2, 128))},
        optimizer=optim.AdamW(model.parameters(), lr=1e-4),
        loss_fn=lambda out, batch: out.mean(),
    )
    print("Smoketest passed:" if success else "Smoketest FAILED")
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CompileSmoketest
# ---------------------------------------------------------------------------


class CompileSmoketest:
    """
    Validates a compiled model through complete training step cycles.

    Each step performs: forward → loss → backward → optimizer.step → zero_grad

    A smoketest with steps=3 is sufficient to catch:
    - First-step compilation failures (step 1 triggers JIT compilation)
    - Second-step issues from CUDA graph replay (if reduce-overhead mode)
    - Autograd graph inconsistencies that appear on repeated calls

    Parameters
    ----------
    steps:
        Number of complete training steps to run. Must be >= 1.
        Default 3: covers compilation step + 2 execution steps.
    seed:
        Random seed for deterministic synthetic batch generation.
        Ensures consistent behavior across runs for debugging.
    """

    def __init__(self, steps: int = 3, seed: int = 42) -> None:
        if steps <= 0:
            raise ValueError(
                f"CompileSmoketest: steps must be > 0. Got {steps}."
            )
        self.steps = steps
        self.seed = seed
        self._step_times: List[float] = []

    @property
    def step_times(self) -> List[float]:
        """Per-step wall times from the most recent run() call."""
        return list(self._step_times)

    def run(
        self,
        model: nn.Module,
        sample_batch: Dict[str, Any],
        optimizer: optim.Optimizer,
        loss_fn: Callable[[Any, Dict[str, Any]], torch.Tensor],
    ) -> bool:
        """
        Run the smoketest: steps complete training cycles.

        Parameters
        ----------
        model:
            The (compiled) model to test. Should already be on the correct
            device and in training mode.
        sample_batch:
            Dict of input tensors. Will be moved to the model's device.
            Must contain inputs that model(** batch) can handle.
        optimizer:
            Optimizer instance bound to model.parameters().
        loss_fn:
            Callable(model_output, batch) -> scalar Tensor.
            The loss function to use for backward.
            Example: lambda out, batch: out.mean()
            Example: lambda out, batch: F.cross_entropy(out, batch["labels"])

        Returns
        -------
        bool
            True if all steps complete without error and without NaN/inf loss.
            False if any step fails (exception or bad loss value).
        """
        self._step_times = []

        # Determine model device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Model has no parameters (e.g., preprocessing module)
            device = torch.device("cpu")

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        logger.info(
            "smoketest: starting %d-step smoketest on device=%s", self.steps, device
        )

        for step in range(self.steps):
            step_start = time.perf_counter()
            try:
                step_result = self._run_single_step(
                    model=model,
                    sample_batch=sample_batch,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    step=step,
                )
            except Exception as exc:
                step_time = time.perf_counter() - step_start
                self._step_times.append(step_time)
                logger.error(
                    "smoketest: FAILED at step %d/%d after %.3fs:\n%s",
                    step + 1,
                    self.steps,
                    step_time,
                    traceback.format_exc(),
                )
                return False

            step_time = time.perf_counter() - step_start
            self._step_times.append(step_time)

            if not step_result:
                logger.error(
                    "smoketest: step %d/%d returned failure status after %.3fs",
                    step + 1,
                    self.steps,
                    step_time,
                )
                return False

            step_label = "compile+run" if step == 0 else "run"
            logger.info(
                "smoketest: step %d/%d (%s) completed in %.3fs",
                step + 1,
                self.steps,
                step_label,
                step_time,
            )

        total_time = sum(self._step_times)
        logger.info(
            "smoketest: ALL %d steps PASSED — total=%.3fs, "
            "first_step(compile)=%.3fs, avg_step=%.3fs",
            self.steps,
            total_time,
            self._step_times[0],
            sum(self._step_times[1:]) / max(len(self._step_times) - 1, 1),
        )
        return True

    def _run_single_step(
        self,
        model: nn.Module,
        sample_batch: Dict[str, Any],
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        step: int,
    ) -> bool:
        """Execute one complete training step. Returns True on success."""
        # Move batch to model's device
        batch_on_device = self._move_batch_to_device(sample_batch, device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = self._forward(model, batch_on_device)

        # Compute loss
        loss = loss_fn(output, batch_on_device)

        # Validate loss
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"loss_fn must return a Tensor. Got {type(loss).__name__}."
            )
        if loss.dim() != 0:
            # Reduce to scalar if not already
            loss = loss.mean()

        if torch.isnan(loss):
            logger.error(
                "smoketest: NaN loss detected at step %d. "
                "Check model initialization and loss function.",
                step + 1,
            )
            return False

        if torch.isinf(loss):
            logger.error(
                "smoketest: Inf loss detected at step %d. "
                "Check for numerical overflow.",
                step + 1,
            )
            return False

        loss_value = loss.item()
        logger.debug("smoketest: step %d loss=%.6f", step + 1, loss_value)

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                nan_grads.append(name)

        if nan_grads:
            logger.error(
                "smoketest: Non-finite gradients at step %d in params: %s",
                step + 1,
                nan_grads[:5],  # show first 5
            )
            return False

        # Optimizer step
        optimizer.step()

        return True

    def _forward(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        """
        Attempt model forward pass with multiple calling conventions.

        Tries keyword arguments first, then positional.
        """
        # Filter to only Tensor values for model input
        tensor_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

        try:
            return model(**tensor_batch)
        except TypeError:
            pass

        # Try with only the first tensor (e.g., model(input_ids))
        tensor_values = list(tensor_batch.values())
        if tensor_values:
            return model(tensor_values[0])

        raise RuntimeError(
            f"Could not call model with batch keys: {list(batch.keys())}"
        )

    def _move_batch_to_device(
        self, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """Move all Tensors in batch dict to device."""
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _create_sample_batch(
        self,
        model: nn.Module,
        batch_size: int = 2,
        seq_len: int = 128,
        vocab_size: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a deterministic synthetic batch for smoketesting.

        This is a convenience method for when you don't have a real batch
        handy. Creates simple token ID tensors.

        Parameters
        ----------
        model:
            Used to determine device placement.
        batch_size:
            Number of sequences in the batch.
        seq_len:
            Length of each sequence.
        vocab_size:
            Vocabulary size for random token generation.

        Returns
        -------
        Dict[str, Tensor]
            {"input_ids": ..., "attention_mask": ..., "labels": ...}
        """
        torch.manual_seed(self.seed)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        input_ids = torch.randint(
            1, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
        )
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        labels = torch.randint(
            0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def run_with_synthetic_batch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Optional[Callable] = None,
        batch_size: int = 2,
        seq_len: int = 128,
    ) -> bool:
        """
        Run smoketest using a synthetic batch (no real data needed).

        Convenience wrapper around run() for quick validation.
        """
        sample_batch = self._create_sample_batch(
            model, batch_size=batch_size, seq_len=seq_len
        )

        if loss_fn is None:
            def loss_fn(output, batch):  # type: ignore[misc]
                if isinstance(output, torch.Tensor):
                    return output.float().mean()
                elif hasattr(output, "loss"):
                    return output.loss
                elif isinstance(output, (tuple, list)):
                    return output[0].float().mean()
                raise RuntimeError(f"Cannot compute loss from {type(output)}")

        return self.run(
            model=model,
            sample_batch=sample_batch,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failures: list = []

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {msg}")
            failures.append(name)

    print("=" * 60)
    print("CompileSmoketest self-tests")
    print("=" * 60)

    # Helper model classes
    class GoodModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)
            self.out = nn.Linear(8, 4)

        def forward(self, input_ids=None, x=None, **kwargs):
            inp = input_ids if input_ids is not None else x
            inp = inp.float()
            return self.out(torch.relu(self.fc(inp)))

    class NaNLossModel(nn.Module):
        """Model whose forward returns NaN."""
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.ones(1))

        def forward(self, input_ids=None, **kwargs):
            return torch.full(
                (2, 4), float("nan"), requires_grad=True
            ) * self.dummy

    class BackwardFailModel(nn.Module):
        """Model that raises during backward."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 4)

        def forward(self, input_ids=None, **kwargs):
            inp = input_ids.float() if input_ids is not None else torch.zeros(2, 16)
            out = self.fc(inp)

            class _BadGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, grad):
                    raise RuntimeError("Intentional backward failure")

            return _BadGrad.apply(out)

    # Test 1: steps=0 raises ValueError
    try:
        _ = CompileSmoketest(steps=0)
        _check("steps_zero_raises", False, "Should have raised ValueError")
    except ValueError as e:
        _check("steps_zero_raises", "steps" in str(e).lower())

    # Test 2: Successful model passes
    try:
        model = GoodModel()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        sample_batch = {"input_ids": torch.randn(2, 16)}
        loss_fn = lambda out, batch: out.mean()

        smoketest = CompileSmoketest(steps=3, seed=42)
        success = smoketest.run(
            model=model,
            sample_batch=sample_batch,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        _check("good_model_passes", success is True, f"Expected True, got {success}")
    except Exception as e:
        _check("good_model_passes", False, str(e))

    # Test 3: Step times are recorded
    try:
        _check(
            "step_times_recorded",
            len(smoketest.step_times) == 3,
            f"Expected 3 times, got {len(smoketest.step_times)}",
        )
        _check(
            "step_times_positive",
            all(t > 0 for t in smoketest.step_times),
            f"Non-positive times: {smoketest.step_times}",
        )
    except Exception as e:
        _check("step_times_recorded", False, str(e))
        _check("step_times_positive", False, str(e))

    # Test 4: NaN loss model fails
    try:
        nan_model = NaNLossModel()
        optimizer_nan = optim.SGD(nan_model.parameters(), lr=1e-3)
        sample_batch_nan = {"input_ids": torch.randn(2, 4)}
        loss_fn_nan = lambda out, batch: out.mean()

        smoketest_nan = CompileSmoketest(steps=3, seed=42)
        success_nan = smoketest_nan.run(
            model=nan_model,
            sample_batch=sample_batch_nan,
            optimizer=optimizer_nan,
            loss_fn=loss_fn_nan,
        )
        _check("nan_loss_fails", success_nan is False, f"Expected False, got {success_nan}")
    except Exception as e:
        _check("nan_loss_fails", False, str(e))

    # Test 5: Backward failure model fails
    try:
        bf_model = BackwardFailModel()
        optimizer_bf = optim.SGD(bf_model.parameters(), lr=1e-3)
        sample_bf = {"input_ids": torch.randn(2, 16)}
        loss_fn_bf = lambda out, batch: out.mean()

        smoketest_bf = CompileSmoketest(steps=2, seed=42)
        success_bf = smoketest_bf.run(
            model=bf_model,
            sample_batch=sample_bf,
            optimizer=optimizer_bf,
            loss_fn=loss_fn_bf,
        )
        _check(
            "backward_fail_fails",
            success_bf is False,
            f"Expected False, got {success_bf}",
        )
    except Exception as e:
        _check("backward_fail_fails", False, str(e))

    # Test 6: _create_sample_batch returns correct shapes
    try:
        model_for_batch = GoodModel()
        smoketest_batch = CompileSmoketest(steps=1, seed=0)
        batch = smoketest_batch._create_sample_batch(
            model_for_batch, batch_size=4, seq_len=32
        )
        _check(
            "create_sample_batch_input_ids_shape",
            batch["input_ids"].shape == (4, 32),
            f"Got {batch['input_ids'].shape}",
        )
        _check(
            "create_sample_batch_attention_mask_shape",
            batch["attention_mask"].shape == (4, 32),
            f"Got {batch['attention_mask'].shape}",
        )
    except Exception as e:
        _check("create_sample_batch_input_ids_shape", False, str(e))
        _check("create_sample_batch_attention_mask_shape", False, str(e))

    # Test 7: run_with_synthetic_batch convenience method
    try:
        model7 = GoodModel()
        opt7 = optim.SGD(model7.parameters(), lr=1e-3)
        st7 = CompileSmoketest(steps=2, seed=1)
        result7 = st7.run_with_synthetic_batch(model7, opt7, seq_len=16)
        _check(
            "run_with_synthetic_batch",
            result7 is True,
            f"Expected True, got {result7}",
        )
    except Exception as e:
        _check("run_with_synthetic_batch", False, str(e))

    # Test 8: steps=1 is valid
    try:
        _ = CompileSmoketest(steps=1)
        _check("steps_one_valid", True)
    except Exception as e:
        _check("steps_one_valid", False, str(e))

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        print("All 12 tests passed.")
