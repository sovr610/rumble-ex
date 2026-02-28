# Eval Harness Reference — Phase 3

## Overview

Phase 3 implements a small, deterministic, quality evaluation harness that runs in under 5 minutes and produces machine-readable metrics. The harness gates on perplexity on a fixed text shard and on task probe accuracy. It is designed to detect quality regressions caused by training changes, architectural modifications, or optimizer bugs—not to serve as a comprehensive benchmark suite.

**Core design constraints:**
- Fixed shard bundled in-repo (no network calls, no dataset drift)
- Fixed tokenizer + fixed truncation rules
- Greedy decode (temperature=0), fixed seeds everywhere
- No LLM-judge in CI (non-deterministic, slow, adds dependency)
- Scored with exact-match or regex only

---

## Determinism Requirements

### Why Determinism Matters

A CI system that gives different results on the same commit is worse than useless—it creates alert fatigue, and false passes miss regressions. Every source of nondeterminism must be eliminated.

**Sources of nondeterminism in inference:**
1. Python `random` module (affects data shuffling if any)
2. NumPy random state (affects any numpy-based sampling)
3. PyTorch random state (affects dropout, data augmentation)
4. CUDA random state (GPU kernels can have internal RNG state)
5. Sampling temperature > 0 (produces different tokens each call)
6. `torch.backends.cudnn.benchmark = True` (algorithm selection changes between runs)

### `set_deterministic_seeds()` Implementation

```python
import random
import os
import numpy as np
import torch

def set_deterministic_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for fully deterministic evaluation.
    Must be called before model loading and before each eval run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Disable cuDNN algorithm auto-tuning (can change between runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable PyTorch deterministic operations where available
    # Note: some ops have no deterministic implementation and will raise RuntimeError
    # Use warn_only=True to avoid breaking on those ops
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass  # PyTorch < 1.8
```

**Call order:**
```python
set_deterministic_seeds(config.seed)
model = load_model(checkpoint_path)
model.train(False)  # model.eval()
result = eval_harness.run()
```

### Validating Determinism

Always verify that running the eval twice on the same checkpoint produces identical results:

```python
def validate_determinism(eval_fn, config, n_runs=2):
    results = []
    for _ in range(n_runs):
        set_deterministic_seeds(config.seed)
        results.append(eval_fn(config))
    assert results[0]["ppl_fixed_shard"] == results[1]["ppl_fixed_shard"], (
        f"Non-deterministic perplexity: "
        f"{results[0]['ppl_fixed_shard']} vs {results[1]['ppl_fixed_shard']}"
    )
    for probe_name in results[0]["task_probe_accuracy"]:
        v0 = results[0]["task_probe_accuracy"][probe_name]
        v1 = results[1]["task_probe_accuracy"][probe_name]
        assert v0 == v1, (
            f"Non-deterministic probe {probe_name}: {v0} vs {v1}"
        )
```

---

## Fixed-Shard Perplexity

### What Perplexity Measures

Perplexity (PPL) measures how surprised the model is by a fixed text sample. Lower is better—a model that assigns high probability to the correct next token has low perplexity.

```
PPL = exp(mean cross-entropy loss over all tokens)
```

For a fixed text of T tokens:
```
CE_loss = -1/T * sum(log P(token_i | token_0..token_{i-1}))
PPL = exp(CE_loss)
```

### Fixed Shard Design

The fixed shard is a small (~50KB) text file bundled into the repository at `bench/data/fixed_shard.txt`. Requirements:

1. **Fixed content**: Never changes after initial creation. Git-tracked.
2. **Diverse**: Covers prose, technical text, code, and structured text for broad language modeling signal.
3. **License-clear**: Use license-permissive text (Wikipedia extracts, permissive-licensed books, synthetic text).
4. **Small**: Should fit in GPU memory easily; tokenized length < 32K tokens.

Example composition (2000-4000 words):
- 40%: Wikipedia-style expository prose on varied topics
- 30%: Technical documentation (code-adjacent)
- 20%: Structured text (lists, tables, Q&A)
- 10%: Synthetic narrative text

### Fixed Truncation Rules

Deterministic truncation avoids off-by-one inconsistencies across runs:

```python
def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    max_length: int = 2048,
    stride: int = 512,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on fixed text with sliding-window approach.
    Uses stride to handle texts longer than max_length without truncation.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # tokens to score in this window
        input_ids_window = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_window.clone()
        target_ids[:, :-trg_len] = -100  # Mask context tokens, score only trg_len

        with torch.no_grad():
            outputs = model(input_ids_window, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return float(ppl.cpu())
```

**Key decisions:**
- `stride < max_length`: Context window slides with overlap so no token is predicted without context
- `target_ids[:, :-trg_len] = -100`: Standard HuggingFace convention for masking loss on context tokens
- `torch.no_grad()`: Disables gradient tracking; this eval should not affect model state
- Stride of 512 with max_length 2048 means 1536-token overlap per window

### Simple Mode (No Sliding Window)

For models with context window >= shard length, simpler:

```python
def compute_perplexity_simple(
    model,
    tokenizer,
    text: str,
    max_length: int = 2048,
    device: str = "cuda",
) -> float:
    encodings = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,  # CRITICAL: must be True and consistent
    )
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(torch.exp(loss).cpu())
```

**Truncation must be explicitly set to `True` with a fixed `max_length`.** Allowing HuggingFace to auto-truncate based on model config can produce different results if the config changes.

---

## Task Probes

### Design Principles

Task probes are small, closed-form tests scored without an LLM judge. Each probe:
1. Has a fixed prompt (few-shot or zero-shot)
2. Has a fixed expected output (exact string or regex)
3. Requires only a short model response (1-20 tokens)
4. Is stable—the correct answer cannot change

### Scoring Methods

**Exact match:**
```python
def score_exact_match(response: str, expected: str) -> bool:
    return response.strip().lower() == expected.strip().lower()
```

**Regex match:**
```python
import re
def score_regex(response: str, pattern: str) -> bool:
    return bool(re.search(pattern, response, re.IGNORECASE))
```

**Prefix match (for models that over-generate):**
```python
def score_prefix(response: str, expected: str) -> bool:
    return response.strip().lower().startswith(expected.strip().lower())
```

### Built-in Probe Definitions

#### `basic_reasoning_25` — 25 QA pairs

Simple factual and arithmetic questions with deterministic correct answers. Examples:

```python
BASIC_REASONING_PROBES = [
    # Arithmetic
    {"prompt": "Q: What is 7 + 8?\nA:", "expected": "15", "method": "prefix"},
    {"prompt": "Q: What is 12 * 4?\nA:", "expected": "48", "method": "prefix"},
    {"prompt": "Q: What is 100 - 37?\nA:", "expected": "63", "method": "prefix"},
    {"prompt": "Q: What is 144 / 12?\nA:", "expected": "12", "method": "prefix"},
    # Factual (stable facts)
    {"prompt": "Q: How many days are in a week?\nA:", "expected": "7", "method": "prefix"},
    {"prompt": "Q: How many months in a year?\nA:", "expected": "12", "method": "prefix"},
    {"prompt": "Q: How many sides does a triangle have?\nA:", "expected": "3", "method": "prefix"},
    {"prompt": "Q: What is the square root of 64?\nA:", "expected": "8", "method": "prefix"},
    # Logic
    {
        "prompt": "Q: If all cats are animals and Whiskers is a cat, is Whiskers an animal? Yes or No.\nA:",
        "expected": "yes",
        "method": "prefix",
    },
    {"prompt": "Q: True or False: 2 + 2 = 5\nA:", "expected": "false", "method": "prefix"},
    # ... 15 more similar items
]
```

Target accuracy for a well-functioning 7B model: > 0.90. Drop below 0.88 triggers FAIL.

#### `format_following_30` — 30 format checks

Tests that the model follows structural formatting instructions. Examples:

```python
FORMAT_FOLLOWING_PROBES = [
    # JSON output
    {
        "prompt": (
            'Output a JSON object with key "status" and value "ok". '
            "Output only the JSON, nothing else."
        ),
        "expected": r'\{"status":\s*"ok"\}',
        "method": "regex",
    },
    # List formatting
    {
        "prompt": "List exactly 3 colors, one per line. Output only the list.",
        "expected": r".+\n.+\n.+",
        "method": "regex",
    },
    # Numeric output
    {
        "prompt": "Output only the number 42.",
        "expected": "42",
        "method": "exact",
    },
    # Capitalization
    {
        "prompt": 'Output the word "python" in all capital letters.',
        "expected": "PYTHON",
        "method": "prefix",
    },
    # ... 26 more
]
```

#### `code_sanity_20` — 20 code completion checks

Tests that the model generates syntactically valid code completions. Examples:

```python
CODE_SANITY_PROBES = [
    # Python function completion
    {
        "prompt": "Complete the Python function:\ndef add(a, b):\n    return",
        "expected": r"return\s+a\s*\+\s*b",
        "method": "regex",
    },
    # Python keyword
    {
        "prompt": "What keyword starts a Python function definition?",
        "expected": "def",
        "method": "prefix",
    },
    # Import syntax
    {
        "prompt": "Complete: import",
        "expected": r"import\s+\w+",
        "method": "regex",
    },
    # ... 17 more
]
```

### Probe Execution

```python
def run_probe(
    model,
    tokenizer,
    probe_def: dict,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> bool:
    """Run a single probe and return True if correct."""
    inputs = tokenizer(
        probe_def["prompt"],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,          # Greedy decode
            do_sample=False,          # Deterministic
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    method = probe_def.get("method", "exact")
    expected = probe_def["expected"]

    if method == "exact":
        return score_exact_match(response, expected)
    elif method == "regex":
        return score_regex(response, expected)
    elif method == "prefix":
        return score_prefix(response, expected)
    else:
        raise ValueError(f"Unknown scoring method: {method}")


def run_probe_set(model, tokenizer, probe_name: str) -> float:
    """Run all probes in a set and return accuracy [0, 1]."""
    probes = PROBE_REGISTRY[probe_name]
    correct = sum(
        1 for probe in probes if run_probe(model, tokenizer, probe)
    )
    return correct / len(probes)
```

### Versioning Probes with Code

Probes are versioned alongside the code. If a probe definition changes, it must be treated as a new probe (rename it), and the baseline for the old probe is discarded. This prevents silent baseline invalidation.

---

## eval.json Schema

```json
{
  "schema_version": "1.0",
  "machine_profile": "H100x8_driver550_cuda12.4_torch2.4_sm90",
  "eval_config": {
    "fixed_shard_path": "bench/data/fixed_shard.txt",
    "fixed_shard_sha256": "a3f2b1...",
    "max_length": 2048,
    "stride": 512,
    "probes": ["basic_reasoning_25", "format_following_30", "code_sanity_20"],
    "temperature": 0.0,
    "seed": 42,
    "max_new_tokens": 32
  },
  "ppl_fixed_shard": 12.34,
  "task_probe_accuracy": {
    "basic_reasoning_25": 0.92,
    "format_following_30": 0.867,
    "code_sanity_20": 0.90
  },
  "env": {
    "torch_version": "2.4.0",
    "cuda_version": "12.4",
    "python_version": "3.11.4",
    "git_sha": "abc1234",
    "git_dirty": false
  },
  "timestamp": "2026-02-21T00:00:00Z"
}
```

**Required fields:**
- `schema_version`: Allows migration if schema changes
- `machine_profile`: Links eval to the hardware context
- `ppl_fixed_shard`: Primary quality metric (float, > 0)
- `task_probe_accuracy`: Dict of `probe_name -> accuracy in [0, 1]`
- `eval_config.fixed_shard_sha256`: Verifies the shard file has not changed

---

## Seed Management

### Complete Seed Setup Pattern

```python
class EvalSmall:
    def __init__(self, config):
        self.config = config

    def run(self, model, tokenizer):
        # Always reset seeds at the start of eval
        set_deterministic_seeds(self.config.seed)

        ppl = compute_perplexity(model, tokenizer, self._load_shard())

        probe_accuracies = {}
        for probe_name in self.config.probes:
            # Reset seeds before each probe set for inter-probe independence
            set_deterministic_seeds(self.config.seed)
            probe_accuracies[probe_name] = run_probe_set(model, tokenizer, probe_name)

        return {"ppl_fixed_shard": ppl, "task_probe_accuracy": probe_accuracies}
```

**Note:** Resetting seeds before each probe set ensures that adding a new probe does not change the seed state for subsequent probes.

### Shard File Integrity Check

```python
import hashlib

def verify_shard(path: str, expected_sha256: str) -> None:
    """Verify the fixed shard has not been modified."""
    with open(path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Fixed shard {path} has unexpected checksum.\n"
            f"Expected: {expected_sha256}\n"
            f"Got: {actual}\n"
            "If you intentionally changed the shard, update the sha256 in EvalConfig."
        )
```

---

## Why No LLM-Judge in CI

LLM-judge evaluation (using a language model to score another model's output) is attractive for open-ended quality assessment but problematic for CI:

| Issue | Details |
|-------|---------|
| **Non-determinism** | Even with temperature=0, judge models may change across API versions |
| **Latency** | API calls add minutes; local judge requires another large model |
| **API dependency** | CI fails if judge API is down |
| **Circular dependency** | Judging brain_ai with brain_ai is circular; external judge adds coupling |
| **Cost** | Hundreds of CI runs per week * judge API cost |
| **Opaque failures** | "Judge says it's worse" gives no actionable signal |

**Better alternatives for CI:**
- Exact-match and regex probes (deterministic, fast, transparent)
- Perplexity on fixed shard (fast, objective)
- Task accuracy on static benchmarks (HellaSwag, MMLU subsets) with fixed answer sets

LLM-judge belongs in periodic offline evaluations, not in every PR gate.

---

## Implementation Checklist

- [ ] `set_deterministic_seeds()` called at start of eval and before each probe set
- [ ] `temperature=0.0` and `do_sample=False` in all `generate()` calls
- [ ] `torch.no_grad()` used in all forward passes
- [ ] Fixed shard path is hardcoded (not configurable without review)
- [ ] Fixed shard SHA256 verified before use
- [ ] `truncation=True` with fixed `max_length` in all tokenizer calls
- [ ] Probe definitions are immutable (changes require new probe name)
- [ ] eval.json written atomically
- [ ] Determinism test passes (same output on two consecutive runs)
- [ ] No network calls during eval
