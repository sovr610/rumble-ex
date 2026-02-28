"""
eval_small_template.py
=======================
EvalSmall class for deterministic, fixed-shard quality evaluation.

Computes:
  - Perplexity on a bundled fixed text shard
  - Accuracy on 3 task probe sets: basic_reasoning_25, format_following_30, code_sanity_20

Key properties:
  - Fully deterministic (fixed seeds, greedy decode, no network calls)
  - Fast enough for CI (under 5 minutes on any modern GPU)
  - No LLM-judge dependency

Usage:
    from eval_small_template import EvalSmall, set_deterministic_seeds
    from perf_gate_config_template import EvalConfig

    config = EvalConfig(seed=42)
    harness = EvalSmall(config, machine_profile="H100x1_...")
    result = harness.run(model, tokenizer)
    harness.save("artifacts/eval.json")

Self-test:
    python eval_small_template.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from perf_gate_config_template import EvalConfig, EvalResult
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class EvalConfig:
        fixed_shard_path: str = "bench/data/fixed_shard.txt"
        fixed_shard_sha256: str = ""
        probes: List[str] = field(default_factory=lambda: [
            "basic_reasoning_25", "format_following_30", "code_sanity_20"
        ])
        temperature: float = 0.0
        seed: int = 42
        max_new_tokens: int = 32
        max_length: int = 2048
        stride: int = 512

        def validate(self):
            pass

    @dataclass
    class EvalResult:
        ppl_fixed_shard: float
        task_probe_accuracy: Dict[str, float]
        machine_profile: str
        eval_config: dict

        def to_dict(self):
            import time as _t
            return {
                "schema_version": "1.0",
                "machine_profile": self.machine_profile,
                "eval_config": self.eval_config,
                "ppl_fixed_shard": self.ppl_fixed_shard,
                "task_probe_accuracy": self.task_probe_accuracy,
                "timestamp": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
            }


# ===========================================================================
# Determinism utilities
# ===========================================================================

def set_deterministic_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for fully deterministic evaluation.

    Must be called before model loading and before each eval run.
    Sets: Python random, NumPy, PyTorch CPU, PyTorch CUDA, PYTHONHASHSEED.
    Also disables cuDNN benchmark mode and enables deterministic algorithms.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _NUMPY_AVAILABLE:
        np.random.seed(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # PyTorch < 1.8


def verify_shard(path: str, expected_sha256: str) -> None:
    """
    Verify the fixed shard file has not been modified.

    Args:
        path: Path to shard file.
        expected_sha256: Expected SHA256 hex digest. Pass empty string to skip.

    Raises:
        FileNotFoundError: If shard file does not exist.
        ValueError: If SHA256 does not match expected.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fixed shard not found: {path}")
    if not expected_sha256:
        return  # Skip verification if no expected hash provided
    with open(path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Fixed shard {path} has unexpected checksum.\n"
            f"Expected: {expected_sha256}\n"
            f"Got:      {actual}\n"
            "If you intentionally changed the shard, update fixed_shard_sha256 in EvalConfig."
        )


# ===========================================================================
# Probe scoring functions
# ===========================================================================

def score_exact_match(response: str, expected: str) -> bool:
    """Case-insensitive, stripped exact match."""
    return response.strip().lower() == expected.strip().lower()


def score_prefix(response: str, expected: str) -> bool:
    """Case-insensitive prefix match (response starts with expected after stripping)."""
    return response.strip().lower().startswith(expected.strip().lower())


def score_regex(response: str, pattern: str) -> bool:
    """Case-insensitive regex search anywhere in response."""
    return bool(re.search(pattern, response, re.IGNORECASE | re.DOTALL))


# ===========================================================================
# Built-in probe definitions
# ===========================================================================

BASIC_REASONING_PROBES = [
    # Arithmetic
    {"prompt": "Q: What is 7 + 8?\nA:", "expected": "15", "method": "prefix"},
    {"prompt": "Q: What is 12 * 4?\nA:", "expected": "48", "method": "prefix"},
    {"prompt": "Q: What is 100 - 37?\nA:", "expected": "63", "method": "prefix"},
    {"prompt": "Q: What is 144 / 12?\nA:", "expected": "12", "method": "prefix"},
    {"prompt": "Q: What is 9 * 9?\nA:", "expected": "81", "method": "prefix"},
    {"prompt": "Q: What is 50 + 50?\nA:", "expected": "100", "method": "prefix"},
    {"prompt": "Q: What is 200 / 4?\nA:", "expected": "50", "method": "prefix"},
    {"prompt": "Q: What is 15 - 7?\nA:", "expected": "8", "method": "prefix"},
    # Factual
    {"prompt": "Q: How many days are in a week?\nA:", "expected": "7", "method": "prefix"},
    {"prompt": "Q: How many months are in a year?\nA:", "expected": "12", "method": "prefix"},
    {"prompt": "Q: How many sides does a triangle have?\nA:", "expected": "3", "method": "prefix"},
    {"prompt": "Q: What is the square root of 64?\nA:", "expected": "8", "method": "prefix"},
    {"prompt": "Q: How many hours are in a day?\nA:", "expected": "24", "method": "prefix"},
    {"prompt": "Q: How many seconds are in a minute?\nA:", "expected": "60", "method": "prefix"},
    {"prompt": "Q: How many days are in a non-leap year?\nA:", "expected": "365", "method": "prefix"},
    # Logic
    {"prompt": "Q: True or False: 2 + 2 = 5\nA:", "expected": "false", "method": "prefix"},
    {"prompt": "Q: True or False: 10 > 5\nA:", "expected": "true", "method": "prefix"},
    {
        "prompt": (
            "Q: If all cats are animals and Whiskers is a cat, "
            "is Whiskers an animal? Answer Yes or No.\nA:"
        ),
        "expected": "yes",
        "method": "prefix",
    },
    {
        "prompt": "Q: If A = B and B = C, does A = C? Answer Yes or No.\nA:",
        "expected": "yes",
        "method": "prefix",
    },
    {"prompt": "Q: What is larger, 0.9 or 0.1?\nA:", "expected": "0.9", "method": "prefix"},
    # Number ordering
    {
        "prompt": "Q: List these numbers in ascending order: 5, 2, 8, 1. Start with the smallest.\nA:",
        "expected": "1",
        "method": "prefix",
    },
    {"prompt": "Q: What is the next number: 1, 2, 3, 4, ?\nA:", "expected": "5", "method": "prefix"},
    {"prompt": "Q: What is 2 to the power of 3?\nA:", "expected": "8", "method": "prefix"},
    {"prompt": "Q: Is 7 a prime number? Answer Yes or No.\nA:", "expected": "yes", "method": "prefix"},
    {"prompt": "Q: What is 0 * 999999?\nA:", "expected": "0", "method": "prefix"},
]

FORMAT_FOLLOWING_PROBES = [
    {"prompt": "Output only the number 42.", "expected": "42", "method": "exact"},
    {"prompt": "Output only the word: STOP", "expected": "STOP", "method": "exact"},
    {"prompt": "Output only the letter A.", "expected": "A", "method": "prefix"},
    {"prompt": 'Output the word "python" in all capital letters.', "expected": "PYTHON", "method": "prefix"},
    {"prompt": 'Output the word "HELLO" in all lowercase letters.', "expected": "hello", "method": "prefix"},
    {
        "prompt": 'Output a JSON object with exactly one key "ok" and value true. Output only the JSON.',
        "expected": r'[{]["\']?ok["\']?\s*:\s*true[}]',
        "method": "regex",
    },
    {
        "prompt": "List exactly 3 primary colors, one per line, no numbering.",
        "expected": r"\w+\n\w+\n\w+",
        "method": "regex",
    },
    {"prompt": "Repeat the following exactly: hello world", "expected": "hello world", "method": "prefix"},
    {"prompt": "Answer with only Yes or No: Is water wet?", "expected": "yes", "method": "prefix"},
    {"prompt": "Answer with only Yes or No: Is 1 + 1 = 3?", "expected": "no", "method": "prefix"},
    {"prompt": "Output exactly two words: good morning", "expected": "good morning", "method": "prefix"},
    {"prompt": "Write the number 5 as a word.", "expected": "five", "method": "prefix"},
    {"prompt": "Write the number 100 as a word.", "expected": "one hundred", "method": "prefix"},
    {
        "prompt": "Output this sentence with a period at the end: The sky is blue",
        "expected": r"The sky is blue\.",
        "method": "regex",
    },
    {"prompt": "What is the opposite of hot? Answer in one word.", "expected": "cold", "method": "prefix"},
    {
        "prompt": "What is the opposite of big? Answer in one word.",
        "expected": r"small|little|tiny",
        "method": "regex",
    },
    {"prompt": "Write 1/2 as a decimal number.", "expected": "0.5", "method": "prefix"},
    {"prompt": "Convert this to snake_case: HelloWorld", "expected": "hello_world", "method": "prefix"},
    {
        "prompt": "Write a valid ISO date for January 1st, 2024.",
        "expected": r"2024-01-01",
        "method": "regex",
    },
    {"prompt": "Complete: The capital of France is", "expected": "paris", "method": "prefix"},
    {"prompt": "Complete: The largest planet in our solar system is", "expected": "jupiter", "method": "prefix"},
    {
        "prompt": "How many characters are in the word 'hello'? Answer with just the number.",
        "expected": "5",
        "method": "prefix",
    },
    {"prompt": "Is Python a programming language? Answer true or false.", "expected": "true", "method": "prefix"},
    {
        "prompt": "How many centimeters are in a meter? Answer with just the number.",
        "expected": "100",
        "method": "prefix",
    },
    {
        "prompt": "Which is larger, 1 kilogram or 1 gram? Answer in one word.",
        "expected": "kilogram",
        "method": "prefix",
    },
    {
        "prompt": "Which letter comes after D in the alphabet? Answer with just the letter.",
        "expected": "E",
        "method": "prefix",
    },
    {"prompt": "How many minutes are in an hour? Answer with just the number.", "expected": "60", "method": "prefix"},
    {"prompt": "Is a square a rectangle? Answer Yes or No.", "expected": "yes", "method": "prefix"},
    {
        "prompt": "If today is Monday, what day is tomorrow? Answer in one word.",
        "expected": "tuesday",
        "method": "prefix",
    },
    {
        "prompt": "What does HTML stand for? Start your answer with 'HyperText'.",
        "expected": "hypertext",
        "method": "prefix",
    },
]

CODE_SANITY_PROBES = [
    {
        "prompt": "What keyword starts a Python function definition? Answer in one word.",
        "expected": "def",
        "method": "prefix",
    },
    {
        "prompt": "What keyword is used to import a module in Python? Answer in one word.",
        "expected": "import",
        "method": "prefix",
    },
    {"prompt": "What Python keyword creates a class? Answer in one word.", "expected": "class", "method": "prefix"},
    {
        "prompt": "What Python keyword exits a function with a value? Answer in one word.",
        "expected": "return",
        "method": "prefix",
    },
    {
        "prompt": "What Python keyword is used for exception handling (the first part)? Answer in one word.",
        "expected": "try",
        "method": "prefix",
    },
    {
        "prompt": "Complete this Python function header:\ndef add(a, b):",
        "expected": r"def add\(a, b\):",
        "method": "regex",
    },
    {
        "prompt": "Write a Python one-liner that prints 'hello'.",
        "expected": r'print\(["\']hello["\']\)',
        "method": "regex",
    },
    {
        "prompt": "Write a Python list with three elements: 1, 2, 3.",
        "expected": r"\[1,\s*2,\s*3\]",
        "method": "regex",
    },
    {
        "prompt": "Write a Python dict with key 'x' and value 1.",
        "expected": r'[{]["\']?x["\']?\s*:\s*1[}]',
        "method": "regex",
    },
    {
        "prompt": "In Python, how do you check if a variable x is None? Complete: x",
        "expected": r"x\s+is\s+None",
        "method": "regex",
    },
    {
        "prompt": "In Python, what does len([1,2,3]) return? Answer with just the number.",
        "expected": "3",
        "method": "prefix",
    },
    {
        "prompt": "In Python, what does type(42).__name__ return? Answer in one word.",
        "expected": "int",
        "method": "prefix",
    },
    {
        "prompt": "Is 'my_var' a valid Python variable name? Answer Yes or No.",
        "expected": "yes",
        "method": "prefix",
    },
    {
        "prompt": "Is '2myvar' a valid Python variable name? Answer Yes or No.",
        "expected": "no",
        "method": "prefix",
    },
    {
        "prompt": "In Python, what does 'hello'.upper() return? Answer in one word.",
        "expected": "HELLO",
        "method": "prefix",
    },
    {
        "prompt": "In Python, what does len('abc') return? Answer with just the number.",
        "expected": "3",
        "method": "prefix",
    },
    {
        "prompt": "In Python, is an empty list [] truthy or falsy? Answer in one word.",
        "expected": "falsy",
        "method": "prefix",
    },
    {
        "prompt": "In Python, what is not True? Answer in one word.",
        "expected": "false",
        "method": "prefix",
    },
    {"prompt": "Complete this Python for loop header: for i in", "expected": r"for i in", "method": "regex"},
    {
        "prompt": (
            "In Python, how many spaces are recommended for indentation per level? "
            "Answer with just the number."
        ),
        "expected": "4",
        "method": "prefix",
    },
]

PROBE_REGISTRY: Dict[str, List[dict]] = {
    "basic_reasoning_25": BASIC_REASONING_PROBES,
    "format_following_30": FORMAT_FOLLOWING_PROBES,
    "code_sanity_20": CODE_SANITY_PROBES,
}


# ===========================================================================
# EvalSmall
# ===========================================================================

class EvalSmall:
    """
    Small, deterministic quality evaluation harness.

    Computes perplexity on a fixed text shard and accuracy on task probes.
    All evaluation is fully deterministic: fixed seeds, greedy decode,
    no network calls, no LLM-judge.

    Example:
        harness = EvalSmall(EvalConfig(), machine_profile="H100x1_...")
        result = harness.run(model, tokenizer)
        harness.save("artifacts/eval.json")
    """

    def __init__(
        self,
        config: EvalConfig,
        machine_profile: str = "unknown",
    ) -> None:
        config.validate()
        self.config = config
        self.machine_profile = machine_profile
        self._result: Optional[EvalResult] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
    ) -> EvalResult:
        """
        Run perplexity computation and all configured probes.

        Args:
            model:      HuggingFace-compatible model with .forward() and .generate().
            tokenizer:  HuggingFace tokenizer.
            device:     Target device ("cuda" or "cpu").

        Returns:
            EvalResult with ppl_fixed_shard and task_probe_accuracy.
        """
        # Reset seeds at start of eval
        set_deterministic_seeds(self.config.seed)

        # Set model to evaluation mode
        if hasattr(model, "eval"):
            model.eval()

        # Compute perplexity on fixed shard
        text = self._load_shard()
        ppl = self._compute_perplexity(model, tokenizer, text, device)

        # Run each probe set with fresh seeds for inter-probe independence
        probe_accuracies: Dict[str, float] = {}
        for probe_name in self.config.probes:
            set_deterministic_seeds(self.config.seed)
            probe_accuracies[probe_name] = self._run_probe_set(
                model, tokenizer, probe_name, device
            )

        self._result = EvalResult(
            ppl_fixed_shard=ppl,
            task_probe_accuracy=probe_accuracies,
            machine_profile=self.machine_profile,
            eval_config=self._build_eval_config_dict(),
        )
        return self._result

    def save(self, path: str) -> None:
        """
        Write eval.json atomically to path.

        Args:
            path: Destination file. Parent directories are created.
        """
        if self._result is None:
            raise RuntimeError("Call run() before save()")

        result_dict = self._result.to_dict()
        result_dict["env"] = self._collect_env_summary()

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=dest.parent, suffix=".tmp", prefix=dest.stem
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)
            os.replace(tmp_path, dest)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # -----------------------------------------------------------------------
    # Internal: perplexity
    # -----------------------------------------------------------------------

    def _load_shard(self) -> str:
        """Load and optionally verify the fixed text shard."""
        path = self.config.fixed_shard_path
        if self.config.fixed_shard_sha256:
            verify_shard(path, self.config.fixed_shard_sha256)

        if not os.path.isfile(path):
            return self._generate_synthetic_shard()

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _generate_synthetic_shard(self) -> str:
        """Generate a minimal synthetic text shard when the real one is missing."""
        return (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning models are trained on large datasets to learn patterns. "
            "Neural networks consist of layers of interconnected nodes. "
            "The gradient descent algorithm minimizes the loss function. "
            "Attention mechanisms allow models to focus on relevant parts of the input. "
        ) * 20

    def _compute_perplexity(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        device: str = "cuda",
    ) -> float:
        """
        Compute perplexity using sliding-window approach for long texts.
        Uses fixed max_length and stride from config for deterministic results.
        """
        if not _TORCH_AVAILABLE:
            return 42.0  # Fallback for environments without torch

        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        # Single-pass for short texts
        if seq_len <= self.config.max_length:
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            return float(torch.exp(loss).cpu())

        # Sliding window for long texts
        nlls: List[torch.Tensor] = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.config.stride):
            end_loc = min(begin_loc + self.config.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_window = input_ids[:, begin_loc:end_loc]
            target_ids = input_window.clone()
            target_ids[:, :-trg_len] = -100  # Mask context tokens

            with torch.no_grad():
                outputs = model(input_window, labels=target_ids)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                neg_log_likelihood = loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
        return float(ppl.cpu())

    # -----------------------------------------------------------------------
    # Internal: probe execution
    # -----------------------------------------------------------------------

    def _run_probe_set(
        self,
        model: Any,
        tokenizer: Any,
        probe_name: str,
        device: str = "cuda",
    ) -> float:
        """
        Run all probes in a named set and return accuracy in [0, 1].

        Args:
            probe_name: Key into PROBE_REGISTRY.

        Returns:
            Fraction of probes answered correctly.
        """
        probes = PROBE_REGISTRY.get(probe_name)
        if probes is None:
            raise ValueError(
                f"Unknown probe set '{probe_name}'. "
                f"Available: {list(PROBE_REGISTRY.keys())}"
            )

        correct = sum(
            1
            for probe in probes
            if self._run_single_probe(model, tokenizer, probe, device)
        )
        return correct / len(probes)

    def _run_single_probe(
        self,
        model: Any,
        tokenizer: Any,
        probe_def: dict,
        device: str = "cuda",
    ) -> bool:
        """
        Run one probe and return True if response matches expected.

        Args:
            probe_def: Dict with keys "prompt", "expected", "method".
        """
        if not _TORCH_AVAILABLE:
            return False

        try:
            inputs = tokenizer(
                probe_def["prompt"],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": self.config.max_new_tokens,
                    "do_sample": False,  # Greedy decode
                    "pad_token_id": (
                        tokenizer.eos_token_id
                        if hasattr(tokenizer, "eos_token_id")
                        else 0
                    ),
                }
                # Only pass temperature if doing sampled decode (not needed for greedy)
                output_ids = model.generate(**inputs, **gen_kwargs)

            # Decode only the newly generated tokens (not the prompt)
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0, prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            return self._score_response(response, probe_def)

        except Exception:
            # Any generation error counts as wrong
            return False

    def _score_response(self, response: str, probe_def: dict) -> bool:
        """Apply the probe's scoring method."""
        method = probe_def.get("method", "exact")
        expected = probe_def["expected"]

        if method == "exact":
            return score_exact_match(response, expected)
        elif method == "prefix":
            return score_prefix(response, expected)
        elif method == "regex":
            return score_regex(response, expected)
        else:
            raise ValueError(f"Unknown scoring method: {method}")

    # -----------------------------------------------------------------------
    # Internal: utilities
    # -----------------------------------------------------------------------

    def _build_eval_config_dict(self) -> dict:
        """Build eval config dict for eval.json."""
        shard_sha = ""
        if os.path.isfile(self.config.fixed_shard_path):
            with open(self.config.fixed_shard_path, "rb") as f:
                shard_sha = hashlib.sha256(f.read()).hexdigest()

        return {
            "fixed_shard_path": self.config.fixed_shard_path,
            "fixed_shard_sha256": shard_sha,
            "max_length": self.config.max_length,
            "stride": self.config.stride,
            "probes": list(self.config.probes),
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_new_tokens": self.config.max_new_tokens,
        }

    def _collect_env_summary(self) -> dict:
        """Lightweight env snapshot for eval.json."""
        torch_ver = torch.__version__ if _TORCH_AVAILABLE else "N/A"
        cuda_ver = "N/A"
        if _TORCH_AVAILABLE:
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", "N/A") or "N/A"
        return {
            "torch_version": torch_ver,
            "cuda_version": cuda_ver,
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
        }


# ===========================================================================
# Self-Tests
# ===========================================================================

def _run_self_tests() -> None:
    """Run all self-tests."""
    print("Running eval_small_template self-tests...")
    failures: List[str] = []

    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
        else:
            print(f"  PASS  {name}")

    # --- set_deterministic_seeds: Python random ---
    set_deterministic_seeds(123)
    r1 = random.random()
    set_deterministic_seeds(123)
    r2 = random.random()
    check("set_seeds.python_random_deterministic", r1 == r2, f"{r1} != {r2}")

    if _NUMPY_AVAILABLE:
        set_deterministic_seeds(42)
        n1 = float(np.random.rand())
        set_deterministic_seeds(42)
        n2 = float(np.random.rand())
        check("set_seeds.numpy_deterministic", n1 == n2, f"{n1} != {n2}")

    if _TORCH_AVAILABLE:
        set_deterministic_seeds(42)
        t1 = torch.rand(1).item()
        set_deterministic_seeds(42)
        t2 = torch.rand(1).item()
        check("set_seeds.torch_deterministic", t1 == t2, f"{t1} != {t2}")

    # --- verify_shard raises on bad hash ---
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("test content")
        tmp_path = f.name
    try:
        try:
            verify_shard(tmp_path, "a" * 64)  # Wrong hash
            check("verify_shard.raises_on_bad_hash", False, "Should have raised ValueError")
        except ValueError:
            check("verify_shard.raises_on_bad_hash", True)

        # Correct hash
        with open(tmp_path, "rb") as f:
            correct_hash = hashlib.sha256(f.read()).hexdigest()
        try:
            verify_shard(tmp_path, correct_hash)
            check("verify_shard.passes_correct_hash", True)
        except ValueError as exc:
            check("verify_shard.passes_correct_hash", False, str(exc))

        # Empty hash = skip
        try:
            verify_shard(tmp_path, "")
            check("verify_shard.skips_empty_hash", True)
        except ValueError as exc:
            check("verify_shard.skips_empty_hash", False, str(exc))
    finally:
        os.unlink(tmp_path)

    # --- verify_shard raises FileNotFoundError ---
    try:
        verify_shard("/nonexistent/path/shard.txt", "abc")
        check("verify_shard.raises_on_missing", False, "Should have raised FileNotFoundError")
    except FileNotFoundError:
        check("verify_shard.raises_on_missing", True)

    # --- Scoring functions ---
    check("score_exact.strips_and_lowercases", score_exact_match("  Hello  ", "hello"))
    check("score_exact.no_match", not score_exact_match("world", "hello"))
    check("score_prefix.match", score_prefix("hello world extra", "hello"))
    check("score_prefix.no_match", not score_prefix("world hello", "hello"))
    check("score_regex.match", score_regex("The answer is 42 tokens.", r"\d+"))
    check("score_regex.no_match", not score_regex("no digits here", r"^\d+$"))
    check("score_regex.case_insensitive", score_regex("HELLO WORLD", "hello"))

    # --- Probe registry completeness ---
    check("PROBE_REGISTRY.has_basic_reasoning", "basic_reasoning_25" in PROBE_REGISTRY)
    check("PROBE_REGISTRY.has_format_following", "format_following_30" in PROBE_REGISTRY)
    check("PROBE_REGISTRY.has_code_sanity", "code_sanity_20" in PROBE_REGISTRY)

    for pset_name, probes in PROBE_REGISTRY.items():
        check(f"PROBE_REGISTRY.{pset_name}.nonempty", len(probes) > 0)
        for i, probe in enumerate(probes):
            check(f"PROBE_REGISTRY.{pset_name}[{i}].has_prompt", "prompt" in probe)
            check(f"PROBE_REGISTRY.{pset_name}[{i}].has_expected", "expected" in probe)
            check(f"PROBE_REGISTRY.{pset_name}[{i}].has_method", "method" in probe)
            check(
                f"PROBE_REGISTRY.{pset_name}[{i}].valid_method",
                probe.get("method") in ("exact", "prefix", "regex"),
            )

    # --- EvalSmall with mock model ---
    if _TORCH_AVAILABLE:
        class MockTokenizer:
            eos_token_id = 2

            def __call__(self, text, return_tensors=None, truncation=False, max_length=None, **kw):
                ids = torch.tensor([[1, 2, 3, 4, 5]])
                return type("Enc", (), {
                    "input_ids": ids,
                    "to": lambda self, dev: self,
                })()

            def decode(self, tokens, skip_special_tokens=True):
                return "15"

        class MockModelOutput:
            def __init__(self, loss_val=2.0):
                self.loss = torch.tensor(loss_val)

            def __getitem__(self, key):
                if key == "loss":
                    return self.loss
                raise KeyError(key)

            def __contains__(self, key):
                return key == "loss"

        class MockModel:
            def __call__(self, input_ids=None, labels=None, **kw):
                return MockModelOutput(2.0)

            def generate(self, input_ids=None, **kw):
                return torch.cat([input_ids, torch.tensor([[2]])], dim=1)

            def train(self, mode=True):
                pass

            def eval(self):
                pass

        config = EvalConfig(seed=42, probes=["basic_reasoning_25"])
        harness = EvalSmall(config, machine_profile="test_profile")
        mock_model = MockModel()
        mock_tokenizer = MockTokenizer()

        result1 = harness.run(mock_model, mock_tokenizer, device="cpu")
        check("eval_small.ppl_positive", result1.ppl_fixed_shard > 0)
        check(
            "eval_small.probe_in_range",
            0 <= result1.task_probe_accuracy.get("basic_reasoning_25", -1) <= 1,
        )

        # Determinism check
        harness2 = EvalSmall(config, machine_profile="test_profile")
        set_deterministic_seeds(42)
        result2 = harness2.run(mock_model, mock_tokenizer, device="cpu")
        check(
            "eval_small.ppl_deterministic",
            abs(result1.ppl_fixed_shard - result2.ppl_fixed_shard) < 1e-6,
            f"{result1.ppl_fixed_shard} != {result2.ppl_fixed_shard}",
        )
        check(
            "eval_small.probes_deterministic",
            result1.task_probe_accuracy == result2.task_probe_accuracy,
        )

        # Save / roundtrip
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "eval.json")
            harness.save(save_path)
            check("eval_small.save_creates_file", os.path.isfile(save_path))
            with open(save_path) as f:
                loaded = json.load(f)
            for key in ["schema_version", "machine_profile", "ppl_fixed_shard", "task_probe_accuracy"]:
                check(f"eval_small.eval_json.has_{key}", key in loaded, f"Missing {key}")
            check(
                "eval_small.eval_json.ppl_matches",
                abs(loaded["ppl_fixed_shard"] - result1.ppl_fixed_shard) < 1e-6,
            )
    else:
        print("  SKIP  EvalSmall tests (PyTorch not available)")

    # Summary
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All eval_small_template self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
