# Reproducibility

## Principle

Reproducible ablations and stable "module competition" require more than setting
a seed once. The orchestrator enforces determinism at every level.

## Seed Management

### Global Seed

```python
def set_global_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
```

### Per-Module RNG Streams

Each module gets a deterministic RNG keyed by `(global_seed, module_name)`:

```python
def create_module_rng(global_seed: int, module_name: str, device: torch.device):
    """Create a deterministic RNG for a specific module."""
    module_seed = global_seed + hash(module_name) % (2**32)
    gen = torch.Generator(device=device)
    gen.manual_seed(module_seed)
    return gen
```

Benefits:
- Adding/removing modules doesn't change other modules' RNG streams
- Same seed + same module name → identical behavior
- RNG state is saveable via `gen.get_state()`

## Deterministic Module Execution

### Fixed Modality Order

ALWAYS process modalities in a sorted list, never from dict iteration:

```python
# WRONG: nondeterministic on Python <3.7, fragile even after
for name, encoder in self.encoders.items():
    ...

# RIGHT: deterministic order
for name in sorted(self.encoders.keys()):
    encoder = self.encoders[name]
    ...
```

### Deterministic Top-K Gating

Workspace competition and attention gating must break ties deterministically:

```python
def stable_topk(
    scores: Tensor,  # (B, N)
    k: int,
    tie_breaker_keys: Optional[Tensor] = None,  # (B, N) secondary sort key
) -> Tuple[Tensor, Tensor]:
    """Top-k with deterministic tie-breaking.

    When scores are tied, breaks ties by:
    1. tie_breaker_keys (if provided)
    2. index (lower index wins)

    Returns:
        values: (B, k) top-k scores
        indices: (B, k) indices
    """
    B, N = scores.shape
    if tie_breaker_keys is not None:
        # Composite key: primary score + tiny tiebreaker
        eps = 1e-10
        composite = scores + eps * tie_breaker_keys
    else:
        # Tie-break by index (lower wins) via tiny perturbation
        eps = 1e-10
        idx_tiebreak = torch.arange(N, device=scores.device).unsqueeze(0)
        composite = scores - eps * idx_tiebreak.float() / N

    values, indices = torch.topk(composite, k, dim=-1, sorted=True)
    # Return original scores at selected indices
    original_values = torch.gather(scores, 1, indices)
    return original_values, indices
```

### Consistent Dropout

Pass the per-module RNG generator to all stochastic operations:

```python
# In workspace stage:
def workspace_run(ctx):
    gen = ctx.rngs.get("workspace")
    # Use generator for any sampling
    if self.training:
        mask = torch.bernoulli(torch.full_like(scores, 1 - dropout_rate),
                               generator=gen)
        scores = scores * mask / (1 - dropout_rate)
    ...
```

## CUDA Nondeterminism

Some CUDA operations are inherently nondeterministic:
- `torch.nn.functional.interpolate` (some modes)
- Scatter/gather with repeated indices
- `atomicAdd` in custom kernels

### Mitigation

1. Enable `torch.use_deterministic_algorithms(True, warn_only=True)`
2. Document known nondeterministic paths in the run manifest
3. For critical comparisons (ablations), use CPU or verify CUDA determinism

## Run Manifest

Every run saves a manifest recording everything needed for reproduction:

```python
def create_run_manifest(brain: BrainAI, seed: int) -> Dict:
    """Create a reproducibility manifest for the current run."""
    return {
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "config": asdict(brain.config),
        "deps_report": brain.deps_report(),
        "module_order": [s.name for s in brain.plan.stages],
        "enabled_stages": [s.name for s in brain.plan.stages if s.enabled],
        "parameter_count": sum(p.numel() for p in brain.parameters()),
        "device": str(next(brain.parameters()).device),
        "dtype_policy": {
            "params": "float32",
            "compute": str(brain.config.training.amp_dtype) if brain.config.training.use_amp else "float32",
        },
    }
```

## Dependency Report

```python
def deps_report(self) -> Dict[str, str]:
    """Report which optional dependencies are installed and what fallbacks were used."""
    report = {}

    # ncps (Neural Circuit Policies)
    try:
        import ncps
        report["ncps"] = f"present (v{ncps.__version__})"
    except ImportError:
        report["ncps"] = "missing → GRU fallback for working memory"

    # htm.core
    try:
        import htm.core
        report["htm_core"] = "present"
    except ImportError:
        report["htm_core"] = "missing → PyTorch HTM or LSTM predictor"

    # pymdp
    try:
        import pymdp
        report["pymdp"] = f"present (v{pymdp.__version__})"
    except ImportError:
        report["pymdp"] = "missing → continuous amortized AIF only"

    # learn2learn
    try:
        import learn2learn
        report["learn2learn"] = f"present (v{learn2learn.__version__})"
    except ImportError:
        report["learn2learn"] = "missing → custom MAML implementation"

    return report
```

## Determinism Verification Test

```python
def test_determinism(brain, sample_input, seed=42):
    """Verify forward pass is deterministic with same seed."""
    set_global_seed(seed)
    out1 = brain(sample_input, return_details=True)

    set_global_seed(seed)
    out2 = brain(sample_input, return_details=True)

    assert torch.equal(out1.output, out2.output), "Output not deterministic"
    assert torch.equal(out1.confidence, out2.confidence), "Confidence not deterministic"

    if out1.details and out1.details.workspace:
        assert torch.equal(
            out1.details.workspace["winners"],
            out2.details.workspace["winners"]
        ), "Workspace winners not deterministic"
```
