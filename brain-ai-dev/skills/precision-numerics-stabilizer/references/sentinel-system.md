# Numerics Sentinel System Reference

## Design Principle

Sentinels are **always-on, low-overhead monitors** running at a configurable cadence (default every 50 steps). They detect **leading indicators** of precision failures before the failure cascade reaches parameters. The detection hierarchy is:

```
logits drift → softmax overflow → loss NaN → gradient NaN → weight NaN (irrecoverable)
```

Sentinels catch the first 1-2 stages. If triggered, deeper checks fire immediately (not waiting for the next cadence window). If the deeper checks confirm the anomaly, a snapshot is captured.

**Design contract**: Zero overhead when inactive. Negligible overhead when active (hooks add ~1% wall time with sampling).

---

## C1: Gradient Norm Sentinel

**When**: After `loss.backward()`, before `optimizer.step()`. For fp16 with GradScaler: after `scaler.unscale_(optimizer)`, before `scaler.step()`.

### Global L2 Norm

```python
global_norm = torch.nn.utils.clip_grad_norm_(
    parameters=model.parameters(),
    max_norm=float('inf')   # compute norm without clipping
)
```

Passing `max_norm=inf` computes the global gradient L2 norm without actually clipping any gradients. The return value is the actual norm. This is the standard PyTorch idiom.

**Alternatively** (if you need norm without even the overhead of the clip check):
```python
total_norm_sq = sum(
    p.grad.detach().float().norm(2).item() ** 2
    for p in model.parameters()
    if p.grad is not None
)
global_norm = total_norm_sq ** 0.5
```

### Per-Module Aggregated Norms

Group parameters by their module prefix, then compute per-module aggregate norms.

```python
from collections import defaultdict

module_norms = defaultdict(float)
for name, param in model.named_parameters():
    if param.grad is None:
        continue
    # Extract module prefix: "blocks.2.attn.q_proj" -> "blocks.2.attn"
    prefix = ".".join(name.split(".")[:-1])  # strip parameter name
    grad_norm_sq = param.grad.detach().float().norm(2).item() ** 2
    module_norms[prefix] += grad_norm_sq

# Convert squared sums to L2 norms
per_module = {k: v ** 0.5 for k, v in module_norms.items()}

# Top-k by norm (descending)
top_k = sorted(per_module.items(), key=lambda x: x[1], reverse=True)[:topk]
```

### Max Individual Parameter Norm

```python
max_param_name, max_param_norm = max(
    ((name, param.grad.detach().float().norm(2).item())
     for name, param in model.named_parameters()
     if param.grad is not None),
    key=lambda x: x[1],
    default=("none", 0.0)
)
```

### GradNormReport

```python
@dataclass
class GradNormReport:
    global_norm: float
    per_module_topk: List[Tuple[str, float]]  # (module_prefix, norm)
    max_param_name: str
    max_param_norm: float
    step: int
    is_finite: bool                            # global_norm is not inf/nan
```

---

## C2: Activation and Weight NaN/Inf Sentinel

### Forward Hook Registration

Register `register_forward_hook` on sampled modules to capture their output tensors. Hooks fire during the forward pass.

```python
class ActivationCapture:
    def __init__(self):
        self.outputs = {}   # layer_name -> tensor

    def make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Detach and move to CPU to avoid impacting backward graph
                self.outputs[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                # Some modules return tuples (attention returns (attn_out, attn_weights))
                tensors = [o for o in output if isinstance(o, torch.Tensor)]
                if tensors:
                    self.outputs[name] = tensors[0].detach()
        return hook
```

**Handle multiple outputs**: Modules like `MultiheadAttention` return `(attn_output, attn_weights)`. Hook the first tensor only.

**Handle non-tensor outputs**: Some modules may return dicts or custom objects. Guard with `isinstance` checks.

### Pattern Matching for Layer Names

Config specifies glob patterns like `"blocks.*.attn"`. Match these against `model.named_modules()`:

```python
import fnmatch

def matches_any_pattern(name: str, patterns: Tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

hooks = []
capture = ActivationCapture()
for name, module in model.named_modules():
    if matches_any_pattern(name, cfg.nan_check_sample_layers):
        handle = module.register_forward_hook(capture.make_hook(name))
        hooks.append(handle)
```

**Remove hooks** after monitoring is complete to prevent memory leaks:
```python
for handle in hooks:
    handle.remove()
```

### Block Rotation Strategy

Instead of monitoring all blocks every interval, rotate through a subset to reduce overhead:

```python
# Every interval, check blocks [0, mid, last]
def get_sample_indices(num_blocks: int) -> List[int]:
    if num_blocks <= 3:
        return list(range(num_blocks))
    return [0, num_blocks // 2, num_blocks - 1]
```

On anomaly detection (any non-finite), immediately check **all** blocks:
```python
if any_nonfinite_detected:
    self._run_full_scan()  # check every layer
```

### Per-Layer Check

For each captured activation tensor:

```python
is_finite = torch.isfinite(tensor).all().item()
max_abs = tensor.abs().max().item() if is_finite else float('inf')
mean = tensor.float().mean().item()
std = tensor.float().std().item()
```

### LayerReport and ActivationReport

```python
@dataclass
class LayerReport:
    name: str
    is_finite: bool
    max_abs: float
    mean: float
    std: float

@dataclass
class ActivationReport:
    layer_reports: List[LayerReport]
    any_nonfinite: bool             # True if any layer had NaN/Inf
    first_nonfinite_name: Optional[str]
    step: int
```

---

## C3: Logit Scale Monitoring

**When**: After the model's forward pass produces logits, before loss computation.

### Why Logit Scale Matters

In fp16, `torch.exp()` overflows at input values above approximately 88.7 (since e^88.7 ≈ 3.4e38 ≈ fp16 max, but fp16 max is only 65504, so overflow occurs at exp(input) > 65504, meaning input > ln(65504) ≈ 11.1). However, with softmax, what matters is the relative scale — the logsumexp computation.

For a numerically stable softmax:
```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

After subtracting max, the arguments to exp are all <= 0, so overflow is not the issue. But very large negative values underflow to 0 (ignored), which is acceptable. The issue arises when logits are processed in **non-numerically-stable implementations** that don't subtract max.

**fp16 practical danger**: If logits are large (max_abs > 65 or so), and the softmax implementation is slightly non-standard, or if logits appear in attention scores (Q*K^T / sqrt(d)), exp overflow becomes likely.

**bf16 practical danger**: Much safer due to wider exponent range (same as fp32), but monitoring is still valuable.

### Threshold and Alert Logic

```python
@dataclass
class LogitReport:
    max_abs: float
    std: float
    exceeds_threshold: bool
    consecutive_violations: int
    step: int
```

Alert after `logit_alert_consecutive` (default 3) consecutive steps where `max_abs > logit_max_abs_threshold` (default 80.0):

```python
if logit_report.max_abs > self.cfg.logit_max_abs_threshold:
    self._consecutive_logit_violations += 1
else:
    self._consecutive_logit_violations = 0

if self._consecutive_logit_violations >= self.cfg.logit_alert_consecutive:
    # emit alert, trigger deeper checks
    logger.warning(
        f"Logit overflow risk: max_abs={logit_report.max_abs:.2f} "
        f"for {self._consecutive_logit_violations} consecutive steps. "
        f"Threshold: {self.cfg.logit_max_abs_threshold:.1f}."
    )
```

---

## Sampling Cadence

Default: check every 50 steps. Configurable per sentinel type.

```python
def should_check(self, step: int) -> bool:
    return step % self.cfg.every_n_steps == 0
```

**Triggering immediate full check**: Any anomaly detection within a sentinel should set a flag that causes all sentinels to run at full depth on the next step (or immediately if possible):

```python
if activation_report.any_nonfinite:
    self._force_check_next = True
    self._run_full_scan(step)  # immediate full depth
```

---

## NumericsReport Aggregate

Combine all sentinel reports into a single dataclass per check interval:

```python
@dataclass
class NumericsReport:
    step: int
    timestamp: float                      # time.time()
    grad_norm_report: Optional[GradNormReport]
    activation_report: Optional[ActivationReport]
    logit_report: Optional[LogitReport]
    weight_report: Optional[WeightReport]

    def to_dict(self) -> dict:
        ...  # serialize for numerics.json in snapshot

    def any_anomaly(self) -> bool:
        """Returns True if any sentinel found a problem."""
        if self.activation_report and self.activation_report.any_nonfinite:
            return True
        if self.logit_report and self.logit_report.consecutive_violations > 0:
            return True
        if self.weight_report and not self.weight_report.all_finite:
            return True
        if self.grad_norm_report and not self.grad_norm_report.is_finite:
            return True
        return False
```

---

## WeightReport

Periodic check (lower cadence, e.g., every 200 steps or on anomaly) that all parameter tensors are finite:

```python
@dataclass
class WeightReport:
    all_finite: bool
    first_nonfinite_name: Optional[str]
    total_params_checked: int
    step: int
```

Implementation:
```python
def check_weights(self) -> WeightReport:
    first_nonfinite = None
    checked = 0
    for name, param in self.model.named_parameters():
        if not torch.isfinite(param.data).all():
            if first_nonfinite is None:
                first_nonfinite = name
        checked += 1
    return WeightReport(
        all_finite=(first_nonfinite is None),
        first_nonfinite_name=first_nonfinite,
        total_params_checked=checked,
        step=step,
    )
```

Weight non-finiteness is the **worst case** — it means parameter corruption has already occurred. Immediate abort is warranted.

---

## Report Format

Logging output format for each check interval:

```
[Step 1000] NumericsMonitor:
  GradNorm: global=0.342, max_param=blocks.11.attn.out_proj.weight (0.891)
  Top-3 modules: blocks.11.attn=0.874, blocks.10.mlp=0.321, blocks.9.attn=0.187
  Activations: all finite. max_abs=12.4 (blocks.5.attn)
  Logits: max_abs=23.7, std=4.2, threshold=80.0, consec_violations=0
  Weights: all finite. 432 tensors checked.
  ScalerScale: 131072.0, skip_rate=0.8% (4/500)
```

Alert format (anomaly detected):
```
[ALERT Step 1050] NumericsMonitor: NON-FINITE ACTIVATION in blocks.8.attn
  is_finite=False, triggering full scan...
  Full scan: 1 non-finite layer found. Capturing snapshot.
  Snapshot written to: runs/run_001/numerics/snapshot_1050/
```
