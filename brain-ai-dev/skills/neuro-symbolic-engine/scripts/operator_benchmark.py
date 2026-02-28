#!/usr/bin/env python3
"""Benchmark suite for the Neuro-Symbolic Engine operators.

Measures:
- Operator throughput (ops/sec) for all fuzzy operator bundles
- Gradient computation time per operator
- Quantifier aggregation throughput at various instance counts
- AMP compatibility and overhead
- Rule evaluation throughput at various rule counts
- End-to-end symbolic reasoning latency
- Memory usage profiling

Usage:
    python operator_benchmark.py                     # All benchmarks
    python operator_benchmark.py --suite operators   # Specific suite
    python operator_benchmark.py --device cuda        # GPU benchmarks
    python operator_benchmark.py --output results.json  # Save results
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Inline Operator Implementations (self-contained, no brain_ai dependency)
# ============================================================================
# These are minimal reproductions of the fuzzy operators from
# brain_ai/reasoning/fuzzy_operators.py, inlined here so the benchmark
# script can run standalone without importing the main package.

# ---------------------------------------------------------------------------
# T-Norms (AND)
# ---------------------------------------------------------------------------

def godel_and(x: Tensor, y: Tensor) -> Tensor:
    """Godel t-norm: min(x, y)."""
    return torch.min(x, y)


def product_and(x: Tensor, y: Tensor) -> Tensor:
    """Product t-norm: x * y."""
    return x * y


def lukasiewicz_and(x: Tensor, y: Tensor) -> Tensor:
    """Lukasiewicz t-norm: max(0, x + y - 1)."""
    return torch.clamp(x + y - 1.0, min=0.0)


def stable_product_and(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    """Stable product t-norm with eps-projection."""
    return torch.clamp(x, min=eps) * torch.clamp(y, min=eps)


# ---------------------------------------------------------------------------
# T-Conorms (OR)
# ---------------------------------------------------------------------------

def godel_or(x: Tensor, y: Tensor) -> Tensor:
    """Godel t-conorm: max(x, y)."""
    return torch.max(x, y)


def product_or(x: Tensor, y: Tensor) -> Tensor:
    """Product t-conorm: x + y - x*y."""
    return x + y - x * y


def lukasiewicz_or(x: Tensor, y: Tensor) -> Tensor:
    """Lukasiewicz t-conorm: min(1, x + y)."""
    return torch.clamp(x + y, max=1.0)


def stable_product_or(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    """Stable product t-conorm with eps-projection."""
    px = torch.clamp(x, max=1.0 - eps)
    py = torch.clamp(y, max=1.0 - eps)
    return 1.0 - (1.0 - px) * (1.0 - py)


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

def standard_negation(x: Tensor) -> Tensor:
    """Standard fuzzy negation: 1 - x."""
    return 1.0 - x


# ---------------------------------------------------------------------------
# Implication
# ---------------------------------------------------------------------------

def reichenbach_implies(x: Tensor, y: Tensor) -> Tensor:
    """Reichenbach S-implication: 1 - x + x*y."""
    return 1.0 - x + x * y


def godel_implies(x: Tensor, y: Tensor) -> Tensor:
    """Godel residuated implication."""
    return torch.where(x <= y, torch.ones_like(x), y)


def lukasiewicz_implies(x: Tensor, y: Tensor) -> Tensor:
    """Lukasiewicz residuated implication: min(1, 1 - x + y)."""
    return torch.clamp(1.0 - x + y, max=1.0)


def stable_product_implies(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    """Stable Reichenbach implication using stable product ops."""
    neg_x = 1.0 - x
    xy = stable_product_and(x, y, eps)
    return stable_product_or(neg_x, xy, eps)


# ---------------------------------------------------------------------------
# Quantifier Aggregators
# ---------------------------------------------------------------------------

class ForallAggregator(nn.Module):
    """Generalized-mean universal quantifier (pMeanError)."""

    def __init__(self, p: float = 2.0, temperature: float = 1.0, stable: bool = True):
        super().__init__()
        self.p = p
        self.temperature = temperature
        self.stable = stable

    def forward(self, truth_values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        tv = truth_values * self.temperature
        errors = torch.clamp(1.0 - tv, min=0.0, max=1.0)
        p = self.p

        if mask is not None:
            errors = errors * mask.float()
            counts = mask.float().sum(dim=-1).clamp(min=1.0)
        else:
            counts = torch.tensor(
                truth_values.shape[-1], dtype=truth_values.dtype,
                device=truth_values.device,
            )

        if self.stable and p > 4.0:
            log_errors = torch.log(errors.clamp(min=1e-20))
            scaled = p * log_errors
            if mask is not None:
                scaled = scaled.masked_fill(~mask.bool(), float("-inf"))
            max_scaled = scaled.max(dim=-1, keepdim=True).values.clamp(min=-40.0)
            exp_shifted = torch.exp(scaled - max_scaled)
            if mask is not None:
                exp_shifted = exp_shifted * mask.float()
            log_mean = max_scaled.squeeze(-1) + torch.log(
                (exp_shifted.sum(dim=-1) / counts).clamp(min=1e-20)
            )
            pmean_error = torch.exp(log_mean / p)
        else:
            powered = errors.pow(p)
            pmean_error = (powered.sum(dim=-1) / counts).pow(1.0 / p)

        return torch.clamp(1.0 - pmean_error, min=0.0, max=1.0)


class ExistsAggregator(nn.Module):
    """Generalized-mean existential quantifier."""

    def __init__(self, p: float = 2.0, temperature: float = 1.0, stable: bool = True):
        super().__init__()
        self.p = p
        self.temperature = temperature
        self.stable = stable

    def forward(self, truth_values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        tv = torch.clamp(truth_values * self.temperature, min=0.0, max=1.0)
        p = self.p

        if mask is not None:
            tv_masked = tv * mask.float()
            counts = mask.float().sum(dim=-1).clamp(min=1.0)
        else:
            tv_masked = tv
            counts = torch.tensor(
                truth_values.shape[-1], dtype=truth_values.dtype,
                device=truth_values.device,
            )

        if self.stable and p > 4.0:
            log_tv = torch.log(tv_masked.clamp(min=1e-20))
            scaled = p * log_tv
            if mask is not None:
                scaled = scaled.masked_fill(~mask.bool(), float("-inf"))
            max_scaled = scaled.max(dim=-1, keepdim=True).values.clamp(min=-40.0)
            exp_shifted = torch.exp(scaled - max_scaled)
            if mask is not None:
                exp_shifted = exp_shifted * mask.float()
            log_mean = max_scaled.squeeze(-1) + torch.log(
                (exp_shifted.sum(dim=-1) / counts).clamp(min=1e-20)
            )
            pmean = torch.exp(log_mean / p)
        else:
            powered = tv_masked.pow(p)
            pmean = (powered.sum(dim=-1) / counts).pow(1.0 / p)

        return torch.clamp(pmean, min=0.0, max=1.0)


# ---------------------------------------------------------------------------
# Operator Bundle
# ---------------------------------------------------------------------------

@dataclass
class OperatorBundle:
    """Complete set of fuzzy connectives."""
    name: str
    and_op: Callable
    or_op: Callable
    not_op: Callable
    implies_op: Callable
    forall: ForallAggregator
    exists: ExistsAggregator


def _make_bundle(name: str, p: float = 2.0) -> OperatorBundle:
    """Factory to create an operator bundle by name."""
    bundles = {
        "godel": OperatorBundle(
            name="godel",
            and_op=godel_and, or_op=godel_or,
            not_op=standard_negation, implies_op=godel_implies,
            forall=ForallAggregator(p=p), exists=ExistsAggregator(p=p),
        ),
        "product": OperatorBundle(
            name="product",
            and_op=product_and, or_op=product_or,
            not_op=standard_negation, implies_op=reichenbach_implies,
            forall=ForallAggregator(p=p), exists=ExistsAggregator(p=p),
        ),
        "lukasiewicz": OperatorBundle(
            name="lukasiewicz",
            and_op=lukasiewicz_and, or_op=lukasiewicz_or,
            not_op=standard_negation, implies_op=lukasiewicz_implies,
            forall=ForallAggregator(p=p), exists=ExistsAggregator(p=p),
        ),
        "stable_product": OperatorBundle(
            name="stable_product",
            and_op=stable_product_and, or_op=stable_product_or,
            not_op=standard_negation, implies_op=stable_product_implies,
            forall=ForallAggregator(p=p), exists=ExistsAggregator(p=p),
        ),
    }
    if name not in bundles:
        raise ValueError(f"Unknown bundle '{name}'. Available: {list(bundles.keys())}")
    return bundles[name]


# ---------------------------------------------------------------------------
# Inline Stub Modules for Rule and End-to-End Benchmarks
# ---------------------------------------------------------------------------

class PredicateModule(nn.Module):
    """Unary predicate P(x) -> truth in [0, 1]."""

    def __init__(self, entity_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, entities: Tensor) -> Tensor:
        return self.mlp(entities).squeeze(-1)


class BilinearRelation(nn.Module):
    """Binary relation R(x, y) via bilinear scoring."""

    def __init__(self, entity_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(entity_dim, entity_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, entities: Tensor) -> Tensor:
        Wx = torch.matmul(entities, self.weight)
        scores = torch.bmm(Wx, entities.transpose(1, 2)) + self.bias
        return torch.sigmoid(scores)


class EntityExtractor(nn.Module):
    """Slot-identity entity extractor: workspace slots -> entities."""

    def __init__(self, workspace_dim: int, entity_dim: int):
        super().__init__()
        self.proj = nn.Linear(workspace_dim, entity_dim)
        self.norm = nn.LayerNorm(entity_dim)

    def forward(self, workspace_slots: Tensor) -> Tuple[Tensor, Tensor]:
        B, K, _ = workspace_slots.shape
        entities = self.norm(self.proj(workspace_slots))
        mask = torch.ones(B, K, dtype=torch.bool, device=entities.device)
        return entities, mask


class RuleAttention(nn.Module):
    """Attention-weighted rule application stub."""

    def __init__(self, rule_dim: int, context_dim: int, num_rules: int):
        super().__init__()
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, rule_dim) * 0.02)
        self.attn_proj = nn.Linear(context_dim + rule_dim, 1)
        self.num_rules = num_rules

    def forward(self, context: Tensor) -> Tensor:
        """Return attention weights over rules. context: (B, context_dim)."""
        B = context.shape[0]
        ctx_expanded = context.unsqueeze(1).expand(B, self.num_rules, -1)
        rule_expanded = self.rule_embeddings.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([ctx_expanded, rule_expanded], dim=-1)
        logits = self.attn_proj(combined).squeeze(-1)  # (B, num_rules)
        return F.softmax(logits, dim=-1)


class SymbolicReasoner(nn.Module):
    """Minimal stub SymbolicReasoner for end-to-end benchmarks.

    Mimics the full pipeline: workspace -> entities -> predicates ->
    rule scoring -> constraint loss -> output.
    """

    def __init__(
        self,
        workspace_dim: int = 256,
        entity_dim: int = 64,
        hidden_dim: int = 128,
        num_predicates: int = 4,
        num_relations: int = 2,
        num_rules: int = 8,
        num_entities: int = 8,
    ):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.entity_dim = entity_dim
        self.num_rules = num_rules
        self.num_entities = num_entities

        # Entity extraction
        self.extractor = EntityExtractor(workspace_dim, entity_dim)

        # Predicates
        self.predicates = nn.ModuleList([
            PredicateModule(entity_dim, hidden_dim)
            for _ in range(num_predicates)
        ])

        # Relations
        self.relations = nn.ModuleList([
            BilinearRelation(entity_dim)
            for _ in range(num_relations)
        ])

        # Rule attention
        self.rule_attention = RuleAttention(
            rule_dim=entity_dim, context_dim=entity_dim, num_rules=num_rules,
        )

        # Output projection
        self.output_proj = nn.Linear(entity_dim, entity_dim)

    def forward(
        self,
        workspace_slots: Tensor,
        return_logs: bool = False,
    ) -> Dict[str, Any]:
        """Full symbolic reasoning forward pass.

        Args:
            workspace_slots: (B, K, workspace_dim)
            return_logs: Whether to return detailed logs.

        Returns:
            Dict with 'constraint_loss', 'truth_values', and optionally 'logs'.
        """
        # Stage 1: Entity extraction
        entities, entity_mask = self.extractor(workspace_slots)

        # Stage 2: Predicate grounding
        pred_truths = []
        for pred in self.predicates:
            pred_truths.append(pred(entities))  # (B, N)

        # Stage 3: Relation grounding
        rel_truths = []
        for rel in self.relations:
            rel_truths.append(rel(entities))  # (B, N, N)

        # Stage 4: Rule scoring (simplified)
        # Use mean entity embedding as context for attention
        context = entities.mean(dim=1)  # (B, entity_dim)
        rule_weights = self.rule_attention(context)  # (B, num_rules)

        # Compute violation per rule (stub: use predicate truths with fuzzy ops)
        bundle = _make_bundle("stable_product")
        violations = []
        for r_idx in range(self.num_rules):
            # Approximate a rule: FORALL x: P_i(x) IMPLIES P_j(x)
            p_idx_a = r_idx % len(self.predicates)
            p_idx_b = (r_idx + 1) % len(self.predicates)
            impl_truth = bundle.implies_op(pred_truths[p_idx_a], pred_truths[p_idx_b])
            # Forall aggregation
            if impl_truth.dim() == 2:
                rule_sat = bundle.forall(impl_truth)  # (B,)
            else:
                rule_sat = impl_truth.mean()
            violations.append(1.0 - rule_sat)

        violation_stack = torch.stack(violations, dim=-1)  # (B, num_rules)
        constraint_loss = (rule_weights * violation_stack).sum(dim=-1).mean()

        # Stage 5: Output
        output = self.output_proj(context)

        result: Dict[str, Any] = {
            "constraint_loss": constraint_loss,
            "truth_values": pred_truths[0],
            "output": output,
        }

        if return_logs:
            result["logs"] = {
                "rule_weights": rule_weights.detach(),
                "violations": violation_stack.detach(),
                "pred_truths": [p.detach() for p in pred_truths],
                "rel_truths": [r.detach() for r in rel_truths],
                "entity_mask": entity_mask,
            }

        return result


# ============================================================================
# Benchmark Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    """Container for a single benchmark measurement."""
    suite: str
    name: str
    batch_size: int
    throughput: float          # ops/sec or items/sec
    mean_latency_ms: float
    std_latency_ms: float
    device: str
    extra: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "suite": self.suite,
            "name": self.name,
            "batch_size": self.batch_size,
            "throughput": self.throughput,
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "device": self.device,
        }
        if self.extra is not None:
            d["extra"] = self.extra
        return d


# ============================================================================
# Benchmark Runner
# ============================================================================

class OperatorBenchmark:
    """Main benchmark harness for neuro-symbolic engine operators."""

    SUITE_MAP = {
        "operators": "_suite_operator_throughput",
        "gradients": "_suite_gradient_computation",
        "quantifiers": "_suite_quantifier_aggregation",
        "amp": "_suite_amp_compatibility",
        "rules": "_suite_rule_scoring",
        "e2e": "_suite_end_to_end",
        "memory": "_suite_memory_profiling",
    }

    def __init__(
        self,
        device: str = "cpu",
        warmup: int = 10,
        iterations: int = 100,
    ):
        self.device = torch.device(device)
        self.warmup = warmup
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []

        # Validate CUDA availability
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Timing utility
    # ------------------------------------------------------------------

    def _time_fn(
        self,
        fn: Callable[[], Any],
        warmup: Optional[int] = None,
        iterations: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Time a function and return (mean_ms, std_ms).

        Performs warmup runs, then timed runs. Uses torch.cuda.synchronize()
        when running on CUDA to get accurate timings.

        Args:
            fn: Zero-argument callable to time.
            warmup: Number of warmup iterations (defaults to self.warmup).
            iterations: Number of timed iterations (defaults to self.iterations).

        Returns:
            Tuple of (mean latency in ms, std latency in ms).
        """
        warmup = warmup if warmup is not None else self.warmup
        iterations = iterations if iterations is not None else self.iterations
        is_cuda = self.device.type == "cuda"

        # Warmup
        for _ in range(warmup):
            fn()
            if is_cuda:
                torch.cuda.synchronize()

        # Timed runs
        timings: List[float] = []
        for _ in range(iterations):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # convert to ms

        mean_ms = sum(timings) / len(timings)
        if len(timings) > 1:
            variance = sum((t - mean_ms) ** 2 for t in timings) / (len(timings) - 1)
            std_ms = math.sqrt(variance)
        else:
            std_ms = 0.0

        return mean_ms, std_ms

    # ------------------------------------------------------------------
    # Suite dispatchers
    # ------------------------------------------------------------------

    def run_suite(self, suite: str) -> List[BenchmarkResult]:
        """Run a specific benchmark suite.

        Args:
            suite: Suite name (one of SUITE_MAP keys).

        Returns:
            List of BenchmarkResult from the suite.
        """
        if suite not in self.SUITE_MAP:
            raise ValueError(
                f"Unknown suite '{suite}'. Available: {list(self.SUITE_MAP.keys())}"
            )
        method_name = self.SUITE_MAP[suite]
        method = getattr(self, method_name)
        results = method()
        self.results.extend(results)
        return results

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmark suites.

        Returns:
            Combined list of all BenchmarkResult instances.
        """
        all_results: List[BenchmarkResult] = []
        for suite_name in self.SUITE_MAP:
            print(f"\n{'='*72}")
            print(f"  Suite: {suite_name}")
            print(f"{'='*72}")
            try:
                results = getattr(self, self.SUITE_MAP[suite_name])()
                all_results.extend(results)
                self.results.extend(results)
            except Exception as e:
                print(f"  [ERROR] Suite '{suite_name}' failed: {e}")
        return all_results

    # ==================================================================
    # Suite 1: Operator Throughput
    # ==================================================================

    def _suite_operator_throughput(self) -> List[BenchmarkResult]:
        """Benchmark AND, OR, NOT, IMPLIES throughput for each bundle.

        Tests at batch sizes 1, 32, 128, 512 with N=10000 operations
        per timing iteration.
        """
        results: List[BenchmarkResult] = []
        bundle_names = ["godel", "product", "lukasiewicz", "stable_product"]
        batch_sizes = [1, 32, 128, 512]
        ops_per_iter = 10000

        for bname in bundle_names:
            bundle = _make_bundle(bname)

            for bs in batch_sizes:
                x = torch.rand(bs, device=self.device)
                y = torch.rand(bs, device=self.device)

                # --- AND ---
                def run_and(op=bundle.and_op, _x=x, _y=y):
                    for _ in range(ops_per_iter):
                        op(_x, _y)

                mean_ms, std_ms = self._time_fn(run_and)
                total_ops = ops_per_iter * bs
                throughput = total_ops / (mean_ms / 1000.0)
                results.append(BenchmarkResult(
                    suite="operators", name=f"{bname}/AND",
                    batch_size=bs, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                ))
                print(f"  {bname}/AND  bs={bs:>4d}  "
                      f"{throughput:>12.0f} ops/s  "
                      f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms")

                # --- OR ---
                def run_or(op=bundle.or_op, _x=x, _y=y):
                    for _ in range(ops_per_iter):
                        op(_x, _y)

                mean_ms, std_ms = self._time_fn(run_or)
                throughput = total_ops / (mean_ms / 1000.0)
                results.append(BenchmarkResult(
                    suite="operators", name=f"{bname}/OR",
                    batch_size=bs, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                ))
                print(f"  {bname}/OR   bs={bs:>4d}  "
                      f"{throughput:>12.0f} ops/s  "
                      f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms")

                # --- NOT ---
                def run_not(op=bundle.not_op, _x=x):
                    for _ in range(ops_per_iter):
                        op(_x)

                mean_ms, std_ms = self._time_fn(run_not)
                total_ops_not = ops_per_iter * bs
                throughput = total_ops_not / (mean_ms / 1000.0)
                results.append(BenchmarkResult(
                    suite="operators", name=f"{bname}/NOT",
                    batch_size=bs, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                ))
                print(f"  {bname}/NOT  bs={bs:>4d}  "
                      f"{throughput:>12.0f} ops/s  "
                      f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms")

                # --- IMPLIES ---
                def run_implies(op=bundle.implies_op, _x=x, _y=y):
                    for _ in range(ops_per_iter):
                        op(_x, _y)

                mean_ms, std_ms = self._time_fn(run_implies)
                throughput = total_ops / (mean_ms / 1000.0)
                results.append(BenchmarkResult(
                    suite="operators", name=f"{bname}/IMPLIES",
                    batch_size=bs, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                ))
                print(f"  {bname}/IMP  bs={bs:>4d}  "
                      f"{throughput:>12.0f} ops/s  "
                      f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms")

        return results

    # ==================================================================
    # Suite 2: Gradient Computation
    # ==================================================================

    def _suite_gradient_computation(self) -> List[BenchmarkResult]:
        """Benchmark forward+backward time and gradient statistics.

        Tests gradient flow through AND and IMPLIES for each bundle,
        at batch sizes 32, 128, 512. Reports gradient magnitude stats
        and compares pure vs stable_product timing.
        """
        results: List[BenchmarkResult] = []
        bundle_names = ["godel", "product", "lukasiewicz", "stable_product"]
        batch_sizes = [32, 128, 512]

        for bname in bundle_names:
            bundle = _make_bundle(bname)

            for bs in batch_sizes:
                # --- AND forward+backward ---
                def run_and_grad(op=bundle.and_op, _bs=bs):
                    x = torch.rand(_bs, device=self.device, requires_grad=True)
                    y = torch.rand(_bs, device=self.device, requires_grad=True)
                    out = op(x, y)
                    loss = out.sum()
                    loss.backward()
                    return x.grad, y.grad

                mean_ms, std_ms = self._time_fn(run_and_grad)
                throughput = bs / (mean_ms / 1000.0)

                # Collect gradient statistics from a single run
                x_stat = torch.rand(bs, device=self.device, requires_grad=True)
                y_stat = torch.rand(bs, device=self.device, requires_grad=True)
                out_stat = bundle.and_op(x_stat, y_stat)
                out_stat.sum().backward()
                gx = x_stat.grad
                gy = y_stat.grad
                if gx is not None and gy is not None:
                    all_grads = torch.cat([gx, gy])
                    grad_mean = all_grads.abs().mean().item()
                    grad_std = all_grads.abs().std().item()
                    grad_min = all_grads.abs().min().item()
                    grad_max = all_grads.abs().max().item()
                else:
                    grad_mean = grad_std = grad_min = grad_max = 0.0

                results.append(BenchmarkResult(
                    suite="gradients", name=f"{bname}/AND_fwd_bwd",
                    batch_size=bs, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                    extra={
                        "grad_mean": grad_mean,
                        "grad_std": grad_std,
                        "grad_min": grad_min,
                        "grad_max": grad_max,
                    },
                ))
                print(f"  {bname}/AND_grad  bs={bs:>4d}  "
                      f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms  "
                      f"grad: mean={grad_mean:.4f} std={grad_std:.4f} "
                      f"min={grad_min:.4f} max={grad_max:.4f}")

                # --- IMPLIES forward+backward ---
                def run_implies_grad(op=bundle.implies_op, _bs=bs):
                    x = torch.rand(_bs, device=self.device, requires_grad=True)
                    y = torch.rand(_bs, device=self.device, requires_grad=True)
                    out = op(x, y)
                    loss = out.sum()
                    loss.backward()
                    return x.grad, y.grad

                mean_ms, std_ms = self._time_fn(run_implies_grad)
                throughput = bs / (mean_ms / 1000.0)

                x_stat2 = torch.rand(bs, device=self.device, requires_grad=True)
                y_stat2 = torch.rand(bs, device=self.device, requires_grad=True)
                out_stat2 = bundle.implies_op(x_stat2, y_stat2)
                out_stat2.sum().backward()
                gx2 = x_stat2.grad
                gy2 = y_stat2.grad
                if gx2 is not None and gy2 is not None:
                    all_grads2 = torch.cat([gx2, gy2])
                    grad_mean2 = all_grads2.abs().mean().item()
                    grad_std2 = all_grads2.abs().std().item()
                    grad_min2 = all_grads2.abs().min().item()
                    grad_max2 = all_grads2.abs().max().item()
                else:
                    grad_mean2 = grad_std2 = grad_min2 = grad_max2 = 0.0

                results.append(BenchmarkResult(
                    suite="gradients", name=f"{bname}/IMPLIES_fwd_bwd",
                    batch_size=bs, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                    extra={
                        "grad_mean": grad_mean2,
                        "grad_std": grad_std2,
                        "grad_min": grad_min2,
                        "grad_max": grad_max2,
                    },
                ))
                print(f"  {bname}/IMP_grad  bs={bs:>4d}  "
                      f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms  "
                      f"grad: mean={grad_mean2:.4f} std={grad_std2:.4f} "
                      f"min={grad_min2:.4f} max={grad_max2:.4f}")

        # --- Compare product vs stable_product gradient computation time ---
        print("\n  --- Gradient timing comparison: product vs stable_product ---")
        comparison_bs = 256
        for op_attr, op_label in [("and_op", "AND"), ("implies_op", "IMPLIES")]:
            timings_map: Dict[str, float] = {}
            for bname in ["product", "stable_product"]:
                bundle = _make_bundle(bname)
                op_fn = getattr(bundle, op_attr)

                def run_grad_compare(op=op_fn, _bs=comparison_bs):
                    x = torch.rand(_bs, device=self.device, requires_grad=True)
                    y = torch.rand(_bs, device=self.device, requires_grad=True)
                    out = op(x, y)
                    out.sum().backward()

                mean_ms, std_ms = self._time_fn(run_grad_compare)
                timings_map[bname] = mean_ms

            speedup = timings_map["product"] / max(timings_map["stable_product"], 1e-9)
            print(f"  {op_label}: product={timings_map['product']:.3f}ms  "
                  f"stable_product={timings_map['stable_product']:.3f}ms  "
                  f"ratio={speedup:.2f}x")

            results.append(BenchmarkResult(
                suite="gradients", name=f"comparison/{op_label}_product_vs_stable",
                batch_size=comparison_bs,
                throughput=comparison_bs / (timings_map["stable_product"] / 1000.0),
                mean_latency_ms=timings_map["stable_product"],
                std_latency_ms=0.0,
                device=str(self.device),
                extra={
                    "product_ms": timings_map["product"],
                    "stable_product_ms": timings_map["stable_product"],
                    "ratio": speedup,
                },
            ))

        return results

    # ==================================================================
    # Suite 3: Quantifier Aggregation
    # ==================================================================

    def _suite_quantifier_aggregation(self) -> List[BenchmarkResult]:
        """Benchmark FORALL and EXISTS aggregation throughput.

        Tests instance counts N=10, 50, 100, 500, 1000 and p values
        1, 2, 5, 10. Also measures masking overhead and gradient time.
        """
        results: List[BenchmarkResult] = []
        instance_counts = [10, 50, 100, 500, 1000]
        p_values = [1, 2, 5, 10]
        batch_size = 32

        # --- FORALL throughput at various N ---
        print("  --- FORALL throughput at various N ---")
        for N in instance_counts:
            fa = ForallAggregator(p=2.0).to(self.device)
            tv = torch.rand(batch_size, N, device=self.device)

            def run_forall(agg=fa, data=tv):
                return agg(data)

            mean_ms, std_ms = self._time_fn(run_forall)
            throughput = (batch_size * N) / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                suite="quantifiers", name=f"FORALL/N={N}",
                batch_size=batch_size, throughput=throughput,
                mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                device=str(self.device),
            ))
            print(f"  FORALL  N={N:>5d}  "
                  f"{throughput:>12.0f} items/s  "
                  f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms")

        # --- EXISTS throughput at various N ---
        print("\n  --- EXISTS throughput at various N ---")
        for N in instance_counts:
            ea = ExistsAggregator(p=2.0).to(self.device)
            tv = torch.rand(batch_size, N, device=self.device)

            def run_exists(agg=ea, data=tv):
                return agg(data)

            mean_ms, std_ms = self._time_fn(run_exists)
            throughput = (batch_size * N) / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                suite="quantifiers", name=f"EXISTS/N={N}",
                batch_size=batch_size, throughput=throughput,
                mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                device=str(self.device),
            ))
            print(f"  EXISTS  N={N:>5d}  "
                  f"{throughput:>12.0f} items/s  "
                  f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms")

        # --- Effect of p parameter on throughput ---
        print("\n  --- Effect of p parameter (N=100) ---")
        N_fixed = 100
        for p_val in p_values:
            fa_p = ForallAggregator(p=float(p_val)).to(self.device)
            tv_p = torch.rand(batch_size, N_fixed, device=self.device)

            def run_forall_p(agg=fa_p, data=tv_p):
                return agg(data)

            mean_ms, std_ms = self._time_fn(run_forall_p)
            throughput = (batch_size * N_fixed) / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                suite="quantifiers", name=f"FORALL/p={p_val}",
                batch_size=batch_size, throughput=throughput,
                mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                device=str(self.device),
                extra={"p": float(p_val), "N": N_fixed},
            ))
            print(f"  FORALL  p={p_val:>2d}  "
                  f"{throughput:>12.0f} items/s  "
                  f"{mean_ms:>8.3f} +/- {std_ms:.3f} ms")

        # --- Masking overhead ---
        print("\n  --- Masking overhead (N=100) ---")
        N_mask = 100
        fa_mask = ForallAggregator(p=2.0).to(self.device)
        tv_mask = torch.rand(batch_size, N_mask, device=self.device)
        mask_tensor = torch.ones(batch_size, N_mask, device=self.device)
        # Randomly mask out approximately 20% of positions
        mask_tensor[torch.rand(batch_size, N_mask) < 0.2] = 0.0

        def run_unmasked(agg=fa_mask, data=tv_mask):
            return agg(data)

        def run_masked(agg=fa_mask, data=tv_mask, m=mask_tensor):
            return agg(data, mask=m)

        mean_unmasked, std_unmasked = self._time_fn(run_unmasked)
        mean_masked, std_masked = self._time_fn(run_masked)

        overhead_pct = ((mean_masked - mean_unmasked) / max(mean_unmasked, 1e-9)) * 100.0
        results.append(BenchmarkResult(
            suite="quantifiers", name="FORALL/masked_vs_unmasked",
            batch_size=batch_size,
            throughput=(batch_size * N_mask) / (mean_masked / 1000.0),
            mean_latency_ms=mean_masked, std_latency_ms=std_masked,
            device=str(self.device),
            extra={
                "unmasked_ms": mean_unmasked,
                "masked_ms": mean_masked,
                "overhead_pct": overhead_pct,
            },
        ))
        print(f"  Unmasked: {mean_unmasked:.3f} ms  "
              f"Masked: {mean_masked:.3f} ms  "
              f"Overhead: {overhead_pct:+.1f}%")

        # --- Gradient computation time for quantifiers ---
        print("\n  --- Quantifier gradient time (N=100) ---")
        N_grad = 100
        for q_name, q_cls in [("FORALL", ForallAggregator), ("EXISTS", ExistsAggregator)]:
            agg = q_cls(p=2.0).to(self.device)

            def run_q_grad(a=agg, _bs=batch_size, _n=N_grad):
                tv = torch.rand(_bs, _n, device=self.device, requires_grad=True)
                out = a(tv)
                out.sum().backward()

            mean_ms, std_ms = self._time_fn(run_q_grad)
            throughput = (batch_size * N_grad) / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                suite="quantifiers", name=f"{q_name}/gradient",
                batch_size=batch_size, throughput=throughput,
                mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                device=str(self.device),
            ))
            print(f"  {q_name} gradient: "
                  f"{mean_ms:.3f} +/- {std_ms:.3f} ms  "
                  f"{throughput:.0f} items/s")

        return results

    # ==================================================================
    # Suite 4: AMP Compatibility
    # ==================================================================

    def _suite_amp_compatibility(self) -> List[BenchmarkResult]:
        """Benchmark AMP compatibility, correctness, and overhead.

        Only runs on CUDA. Compares fp32 vs AMP throughput for each
        operator, verifies output correctness, and measures memory savings.
        """
        results: List[BenchmarkResult] = []

        if self.device.type != "cuda":
            print("  [SKIP] AMP benchmarks require CUDA. Run with --device cuda.")
            results.append(BenchmarkResult(
                suite="amp", name="SKIPPED/no_cuda",
                batch_size=0, throughput=0.0,
                mean_latency_ms=0.0, std_latency_ms=0.0,
                device=str(self.device),
                extra={"reason": "CUDA not available"},
            ))
            return results

        bundle_names = ["godel", "product", "lukasiewicz", "stable_product"]
        batch_size = 512
        ops_per_iter = 5000

        for bname in bundle_names:
            bundle = _make_bundle(bname)
            x_fp32 = torch.rand(batch_size, device=self.device)
            y_fp32 = torch.rand(batch_size, device=self.device)

            for op_name, op_attr in [
                ("AND", "and_op"), ("OR", "or_op"),
                ("NOT", "not_op"), ("IMPLIES", "implies_op"),
            ]:
                op_fn = getattr(bundle, op_attr)
                is_unary = op_name == "NOT"

                # --- fp32 baseline ---
                if is_unary:
                    def run_fp32(op=op_fn, _x=x_fp32):
                        for _ in range(ops_per_iter):
                            op(_x)
                else:
                    def run_fp32(op=op_fn, _x=x_fp32, _y=y_fp32):
                        for _ in range(ops_per_iter):
                            op(_x, _y)

                mean_fp32, std_fp32 = self._time_fn(run_fp32)

                # --- AMP ---
                if is_unary:
                    def run_amp(op=op_fn, _x=x_fp32):
                        with torch.cuda.amp.autocast(enabled=True):
                            for _ in range(ops_per_iter):
                                op(_x)
                else:
                    def run_amp(op=op_fn, _x=x_fp32, _y=y_fp32):
                        with torch.cuda.amp.autocast(enabled=True):
                            for _ in range(ops_per_iter):
                                op(_x, _y)

                mean_amp, std_amp = self._time_fn(run_amp)

                # --- Correctness check ---
                with torch.no_grad():
                    if is_unary:
                        out_fp32 = op_fn(x_fp32)
                        with torch.cuda.amp.autocast(enabled=True):
                            out_amp = op_fn(x_fp32)
                    else:
                        out_fp32 = op_fn(x_fp32, y_fp32)
                        with torch.cuda.amp.autocast(enabled=True):
                            out_amp = op_fn(x_fp32, y_fp32)

                output_close = torch.allclose(
                    out_fp32.float(), out_amp.float(), atol=1e-3, rtol=1e-3,
                )

                speedup = mean_fp32 / max(mean_amp, 1e-9)
                total_ops = ops_per_iter * batch_size

                results.append(BenchmarkResult(
                    suite="amp", name=f"{bname}/{op_name}",
                    batch_size=batch_size,
                    throughput=total_ops / (mean_amp / 1000.0),
                    mean_latency_ms=mean_amp, std_latency_ms=std_amp,
                    device=str(self.device),
                    extra={
                        "fp32_ms": mean_fp32,
                        "amp_ms": mean_amp,
                        "speedup": speedup,
                        "output_correct": float(output_close),
                    },
                ))
                correct_str = "OK" if output_close else "MISMATCH"
                print(f"  {bname}/{op_name:>7s}  "
                      f"fp32={mean_fp32:.3f}ms  amp={mean_amp:.3f}ms  "
                      f"speedup={speedup:.2f}x  correct={correct_str}")

        # --- Memory savings under AMP ---
        print("\n  --- AMP memory savings ---")
        try:
            torch.cuda.reset_peak_memory_stats()
            x_mem = torch.rand(10000, device=self.device)
            y_mem = torch.rand(10000, device=self.device)
            for _ in range(1000):
                stable_product_and(x_mem, y_mem)
            torch.cuda.synchronize()
            mem_fp32 = torch.cuda.max_memory_allocated() / (1024 * 1024)

            torch.cuda.reset_peak_memory_stats()
            with torch.cuda.amp.autocast(enabled=True):
                for _ in range(1000):
                    stable_product_and(x_mem, y_mem)
            torch.cuda.synchronize()
            mem_amp = torch.cuda.max_memory_allocated() / (1024 * 1024)

            savings_pct = ((mem_fp32 - mem_amp) / max(mem_fp32, 1e-9)) * 100.0
            results.append(BenchmarkResult(
                suite="amp", name="memory_savings",
                batch_size=10000, throughput=0.0,
                mean_latency_ms=0.0, std_latency_ms=0.0,
                device=str(self.device),
                extra={
                    "fp32_mb": mem_fp32,
                    "amp_mb": mem_amp,
                    "savings_pct": savings_pct,
                },
            ))
            print(f"  fp32: {mem_fp32:.2f} MB  "
                  f"AMP: {mem_amp:.2f} MB  "
                  f"Savings: {savings_pct:.1f}%")
        except Exception as e:
            print(f"  [WARN] Memory profiling failed: {e}")

        return results

    # ==================================================================
    # Suite 5: Rule Scoring
    # ==================================================================

    def _suite_rule_scoring(self) -> List[BenchmarkResult]:
        """Benchmark rule scoring throughput.

        Tests R=1, 5, 10, 25, 50 rules on N=10, 32 entities.
        Measures simple rules (single literal) vs complex rules
        (nested quantifiers), attention computation overhead, and
        constraint loss computation time.
        """
        results: List[BenchmarkResult] = []
        rule_counts = [1, 5, 10, 25, 50]
        entity_counts = [10, 32]
        batch_size = 8
        entity_dim = 64
        hidden_dim = 128

        bundle = _make_bundle("stable_product")

        print("  --- Simple rules (single AND literal) ---")
        for R in rule_counts:
            for N in entity_counts:
                # Create predicate truths
                pred_truths = [
                    torch.rand(batch_size, N, device=self.device)
                    for _ in range(max(R, 2))
                ]

                def run_simple_rules(n_rules=R, truths=pred_truths, _bundle=bundle):
                    violations = []
                    for r in range(n_rules):
                        p_a = truths[r % len(truths)]
                        p_b = truths[(r + 1) % len(truths)]
                        # Simple rule: AND(P_a, P_b)
                        rule_out = _bundle.and_op(p_a, p_b)
                        violation = 1.0 - rule_out.mean(dim=-1)
                        violations.append(violation)
                    return torch.stack(violations, dim=-1)

                mean_ms, std_ms = self._time_fn(run_simple_rules)
                throughput = R / (mean_ms / 1000.0)
                results.append(BenchmarkResult(
                    suite="rules", name=f"simple/R={R}_N={N}",
                    batch_size=batch_size, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                    extra={"num_rules": R, "num_entities": N},
                ))
                print(f"  simple  R={R:>3d}  N={N:>3d}  "
                      f"{throughput:>10.0f} rules/s  "
                      f"{mean_ms:.3f} +/- {std_ms:.3f} ms")

        print("\n  --- Complex rules (nested quantifier + implication) ---")
        for R in rule_counts:
            for N in entity_counts:
                pred_truths = [
                    torch.rand(batch_size, N, device=self.device)
                    for _ in range(max(R, 2))
                ]
                fa = ForallAggregator(p=2.0).to(self.device)

                def run_complex_rules(
                    n_rules=R, truths=pred_truths, forall=fa, _bundle=bundle,
                ):
                    violations = []
                    for r in range(n_rules):
                        p_a = truths[r % len(truths)]
                        p_b = truths[(r + 1) % len(truths)]
                        # Complex: FORALL x: IMPLIES(P_a(x), OR(P_b(x), NOT(P_a(x))))
                        impl_body = _bundle.implies_op(
                            p_a,
                            _bundle.or_op(p_b, _bundle.not_op(p_a)),
                        )
                        rule_sat = forall(impl_body)
                        violations.append(1.0 - rule_sat)
                    return torch.stack(violations, dim=-1)

                mean_ms, std_ms = self._time_fn(run_complex_rules)
                throughput = R / (mean_ms / 1000.0)
                results.append(BenchmarkResult(
                    suite="rules", name=f"complex/R={R}_N={N}",
                    batch_size=batch_size, throughput=throughput,
                    mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                    device=str(self.device),
                    extra={"num_rules": R, "num_entities": N},
                ))
                print(f"  complex R={R:>3d}  N={N:>3d}  "
                      f"{throughput:>10.0f} rules/s  "
                      f"{mean_ms:.3f} +/- {std_ms:.3f} ms")

        # --- Attention computation overhead ---
        print("\n  --- Attention computation overhead ---")
        for R in [10, 25, 50]:
            rule_attn = RuleAttention(
                rule_dim=entity_dim, context_dim=entity_dim, num_rules=R,
            ).to(self.device)
            context = torch.randn(batch_size, entity_dim, device=self.device)

            def run_attn(attn=rule_attn, ctx=context):
                return attn(ctx)

            mean_ms, std_ms = self._time_fn(run_attn)
            throughput = batch_size / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                suite="rules", name=f"attention/R={R}",
                batch_size=batch_size, throughput=throughput,
                mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                device=str(self.device),
                extra={"num_rules": R},
            ))
            print(f"  attention  R={R:>3d}  "
                  f"{mean_ms:.3f} +/- {std_ms:.3f} ms  "
                  f"{throughput:.0f} batch/s")

        # --- Constraint loss computation time ---
        print("\n  --- Constraint loss computation ---")
        for R in rule_counts:
            rule_attn = RuleAttention(
                rule_dim=entity_dim, context_dim=entity_dim, num_rules=R,
            ).to(self.device)
            context = torch.randn(batch_size, entity_dim, device=self.device)
            violations = torch.rand(batch_size, R, device=self.device)

            def run_constraint_loss(attn=rule_attn, ctx=context, viol=violations):
                weights = attn(ctx)
                loss = (weights * viol).sum(dim=-1).mean()
                return loss

            mean_ms, std_ms = self._time_fn(run_constraint_loss)
            throughput = (batch_size * R) / (mean_ms / 1000.0)
            results.append(BenchmarkResult(
                suite="rules", name=f"constraint_loss/R={R}",
                batch_size=batch_size, throughput=throughput,
                mean_latency_ms=mean_ms, std_latency_ms=std_ms,
                device=str(self.device),
                extra={"num_rules": R},
            ))
            print(f"  constraint  R={R:>3d}  "
                  f"{mean_ms:.3f} +/- {std_ms:.3f} ms")

        return results

    # ==================================================================
    # Suite 6: End-to-End Symbolic Reasoning
    # ==================================================================

    def _suite_end_to_end(self) -> List[BenchmarkResult]:
        """Benchmark full SymbolicReasoner forward pass.

        Measures each stage independently and total latency at
        batch sizes 1, 8, 32 with and without return_logs.
        """
        results: List[BenchmarkResult] = []
        batch_sizes = [1, 8, 32]
        workspace_dim = 256
        entity_dim = 64
        hidden_dim = 128
        num_predicates = 4
        num_relations = 2
        num_rules = 8
        num_entities = 8

        reasoner = SymbolicReasoner(
            workspace_dim=workspace_dim,
            entity_dim=entity_dim,
            hidden_dim=hidden_dim,
            num_predicates=num_predicates,
            num_relations=num_relations,
            num_rules=num_rules,
            num_entities=num_entities,
        ).to(self.device)
        reasoner_for_train = SymbolicReasoner(
            workspace_dim=workspace_dim,
            entity_dim=entity_dim,
            hidden_dim=hidden_dim,
            num_predicates=num_predicates,
            num_relations=num_relations,
            num_rules=num_rules,
            num_entities=num_entities,
        ).to(self.device)

        for bs in batch_sizes:
            ws = torch.randn(bs, num_entities, workspace_dim, device=self.device)

            # --- Per-stage timing ---
            print(f"\n  --- Batch size {bs}: per-stage breakdown ---")

            reasoner.train(False)

            # Stage 1: Entity extraction
            def run_extract(data=ws, ext=reasoner.extractor):
                return ext(data)

            mean_extract, std_extract = self._time_fn(run_extract)
            print(f"    Entity extraction:   {mean_extract:.3f} +/- {std_extract:.3f} ms")

            # Stage 2: Predicate grounding
            with torch.no_grad():
                entities, _ = reasoner.extractor(ws)

            def run_predicates(ent=entities, preds=reasoner.predicates):
                return [p(ent) for p in preds]

            mean_pred, std_pred = self._time_fn(run_predicates)
            print(f"    Predicate grounding: {mean_pred:.3f} +/- {std_pred:.3f} ms")

            # Stage 3: Relation grounding
            def run_relations(ent=entities, rels=reasoner.relations):
                return [r(ent) for r in rels]

            mean_rel, std_rel = self._time_fn(run_relations)
            print(f"    Relation grounding:  {mean_rel:.3f} +/- {std_rel:.3f} ms")

            # Stage 4: Rule scoring (attention + violation)
            with torch.no_grad():
                context = entities.mean(dim=1)

            def run_rules(ctx=context, attn=reasoner.rule_attention):
                return attn(ctx)

            mean_rules, std_rules = self._time_fn(run_rules)
            print(f"    Rule attention:      {mean_rules:.3f} +/- {std_rules:.3f} ms")

            # --- Total forward pass (no logs) ---
            def run_total(data=ws, model=reasoner):
                with torch.no_grad():
                    return model(data, return_logs=False)

            mean_total, std_total = self._time_fn(run_total)
            throughput = bs / (mean_total / 1000.0)
            print(f"    TOTAL (no logs):     {mean_total:.3f} +/- {std_total:.3f} ms  "
                  f"({throughput:.0f} samples/s)")

            results.append(BenchmarkResult(
                suite="e2e", name=f"forward/bs={bs}",
                batch_size=bs, throughput=throughput,
                mean_latency_ms=mean_total, std_latency_ms=std_total,
                device=str(self.device),
                extra={
                    "extract_ms": mean_extract,
                    "predicate_ms": mean_pred,
                    "relation_ms": mean_rel,
                    "rules_ms": mean_rules,
                    "total_ms": mean_total,
                },
            ))

            # --- Total forward pass (with logs) ---
            def run_total_logs(data=ws, model=reasoner):
                with torch.no_grad():
                    return model(data, return_logs=True)

            mean_total_logs, std_total_logs = self._time_fn(run_total_logs)
            throughput_logs = bs / (mean_total_logs / 1000.0)
            log_overhead = (
                (mean_total_logs - mean_total) / max(mean_total, 1e-9) * 100.0
            )
            print(f"    TOTAL (with logs):   {mean_total_logs:.3f} +/- {std_total_logs:.3f} ms  "
                  f"({throughput_logs:.0f} samples/s)  "
                  f"log overhead: {log_overhead:+.1f}%")

            results.append(BenchmarkResult(
                suite="e2e", name=f"forward_logs/bs={bs}",
                batch_size=bs, throughput=throughput_logs,
                mean_latency_ms=mean_total_logs, std_latency_ms=std_total_logs,
                device=str(self.device),
                extra={
                    "total_no_logs_ms": mean_total,
                    "total_with_logs_ms": mean_total_logs,
                    "log_overhead_pct": log_overhead,
                },
            ))

        # --- Forward + backward total ---
        print("\n  --- Forward + backward latency ---")
        for bs in batch_sizes:
            ws = torch.randn(bs, num_entities, workspace_dim, device=self.device)
            reasoner_for_train.train(True)

            def run_fwd_bwd(data=ws, model=reasoner_for_train):
                model.zero_grad()
                out = model(data, return_logs=False)
                loss = out["constraint_loss"]
                loss.backward()

            mean_fb, std_fb = self._time_fn(run_fwd_bwd)
            throughput_fb = bs / (mean_fb / 1000.0)
            print(f"    bs={bs:>3d}  fwd+bwd: {mean_fb:.3f} +/- {std_fb:.3f} ms  "
                  f"({throughput_fb:.0f} samples/s)")

            results.append(BenchmarkResult(
                suite="e2e", name=f"fwd_bwd/bs={bs}",
                batch_size=bs, throughput=throughput_fb,
                mean_latency_ms=mean_fb, std_latency_ms=std_fb,
                device=str(self.device),
            ))

        return results

    # ==================================================================
    # Suite 7: Memory Profiling
    # ==================================================================

    def _suite_memory_profiling(self) -> List[BenchmarkResult]:
        """Profile memory usage for operators, entity/rule scaling, and logs.

        Measures:
        - Peak memory per operator bundle at batch_size=128
        - Memory scaling with entity count (N=8, 16, 32, 64)
        - Memory scaling with rule count (R=8, 16, 32, 64)
        - Log overhead when return_logs=True vs False
        """
        results: List[BenchmarkResult] = []
        is_cuda = self.device.type == "cuda"

        def _get_memory_mb() -> float:
            """Return current peak memory in MB (CUDA) or estimate (CPU)."""
            if is_cuda:
                torch.cuda.synchronize()
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                # On CPU, we cannot track peak memory directly.
                # Return 0 and note this in the output.
                return 0.0

        def _reset_memory():
            """Reset memory tracking."""
            gc.collect()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

        # --- Peak memory per operator bundle ---
        print("  --- Peak memory per operator bundle (batch_size=128) ---")
        batch_size = 128
        bundle_names = ["godel", "product", "lukasiewicz", "stable_product"]

        for bname in bundle_names:
            bundle = _make_bundle(bname)
            _reset_memory()

            x = torch.rand(batch_size, device=self.device, requires_grad=True)
            y = torch.rand(batch_size, device=self.device, requires_grad=True)

            # Run all ops
            a = bundle.and_op(x, y)
            o = bundle.or_op(x, y)
            n = bundle.not_op(x)
            i = bundle.implies_op(x, y)
            loss = a.sum() + o.sum() + n.sum() + i.sum()
            loss.backward()

            mem_mb = _get_memory_mb()
            results.append(BenchmarkResult(
                suite="memory", name=f"bundle_peak/{bname}",
                batch_size=batch_size, throughput=0.0,
                mean_latency_ms=0.0, std_latency_ms=0.0,
                device=str(self.device),
                extra={"peak_memory_mb": mem_mb},
            ))
            if is_cuda:
                print(f"  {bname:>16s}: {mem_mb:.2f} MB")
            else:
                print(f"  {bname:>16s}: (CPU - no peak tracking)")

        # --- Memory scaling with entity count ---
        print("\n  --- Memory scaling with entity count ---")
        entity_counts = [8, 16, 32, 64]
        workspace_dim = 256
        entity_dim = 64
        hidden_dim = 128

        for N in entity_counts:
            _reset_memory()

            extractor = EntityExtractor(workspace_dim, entity_dim).to(self.device)
            pred = PredicateModule(entity_dim, hidden_dim).to(self.device)
            rel = BilinearRelation(entity_dim).to(self.device)

            ws = torch.randn(8, N, workspace_dim, device=self.device)
            entities, mask = extractor(ws)
            p_truth = pred(entities)
            r_truth = rel(entities)
            loss = p_truth.sum() + r_truth.sum()
            loss.backward()

            mem_mb = _get_memory_mb()
            results.append(BenchmarkResult(
                suite="memory", name=f"entity_scaling/N={N}",
                batch_size=8, throughput=0.0,
                mean_latency_ms=0.0, std_latency_ms=0.0,
                device=str(self.device),
                extra={"num_entities": N, "peak_memory_mb": mem_mb},
            ))
            if is_cuda:
                print(f"  N={N:>3d}: {mem_mb:.2f} MB")
            else:
                print(f"  N={N:>3d}: (CPU - no peak tracking)")

            del extractor, pred, rel, ws, entities, mask, p_truth, r_truth, loss

        # --- Memory scaling with rule count ---
        print("\n  --- Memory scaling with rule count ---")
        rule_counts = [8, 16, 32, 64]
        batch_size_r = 8
        N_fixed = 16

        for R in rule_counts:
            _reset_memory()

            bundle = _make_bundle("stable_product")
            fa = ForallAggregator(p=2.0).to(self.device)
            rule_attn = RuleAttention(
                rule_dim=entity_dim, context_dim=entity_dim, num_rules=R,
            ).to(self.device)

            pred_truths = [
                torch.rand(batch_size_r, N_fixed, device=self.device, requires_grad=True)
                for _ in range(max(R, 2))
            ]
            context = torch.randn(batch_size_r, entity_dim, device=self.device)

            weights = rule_attn(context)
            violations = []
            for r_idx in range(R):
                p_a = pred_truths[r_idx % len(pred_truths)]
                p_b = pred_truths[(r_idx + 1) % len(pred_truths)]
                impl = bundle.implies_op(p_a, p_b)
                sat = fa(impl)
                violations.append(1.0 - sat)
            v_stack = torch.stack(violations, dim=-1)
            loss = (weights * v_stack).sum(dim=-1).mean()
            loss.backward()

            mem_mb = _get_memory_mb()
            results.append(BenchmarkResult(
                suite="memory", name=f"rule_scaling/R={R}",
                batch_size=batch_size_r, throughput=0.0,
                mean_latency_ms=0.0, std_latency_ms=0.0,
                device=str(self.device),
                extra={"num_rules": R, "peak_memory_mb": mem_mb},
            ))
            if is_cuda:
                print(f"  R={R:>3d}: {mem_mb:.2f} MB")
            else:
                print(f"  R={R:>3d}: (CPU - no peak tracking)")

            del bundle, fa, rule_attn, pred_truths, context, weights, violations
            del v_stack, loss

        # --- Log overhead: return_logs=True vs False ---
        print("\n  --- Log overhead: return_logs=True vs False ---")
        reasoner = SymbolicReasoner(
            workspace_dim=workspace_dim,
            entity_dim=entity_dim,
            hidden_dim=hidden_dim,
            num_predicates=4,
            num_relations=2,
            num_rules=16,
            num_entities=16,
        ).to(self.device)
        reasoner.train(False)

        ws = torch.randn(8, 16, workspace_dim, device=self.device)

        # Without logs
        _reset_memory()
        with torch.no_grad():
            _ = reasoner(ws, return_logs=False)
        mem_no_logs = _get_memory_mb()

        # With logs
        _reset_memory()
        with torch.no_grad():
            _ = reasoner(ws, return_logs=True)
        mem_with_logs = _get_memory_mb()

        overhead = mem_with_logs - mem_no_logs
        results.append(BenchmarkResult(
            suite="memory", name="log_overhead",
            batch_size=8, throughput=0.0,
            mean_latency_ms=0.0, std_latency_ms=0.0,
            device=str(self.device),
            extra={
                "no_logs_mb": mem_no_logs,
                "with_logs_mb": mem_with_logs,
                "overhead_mb": overhead,
            },
        ))
        if is_cuda:
            print(f"  No logs:   {mem_no_logs:.2f} MB")
            print(f"  With logs: {mem_with_logs:.2f} MB")
            print(f"  Overhead:  {overhead:+.2f} MB")
        else:
            print("  (CPU - no peak memory tracking, overhead measured via timing)")

        return results


# ============================================================================
# Output Formatting
# ============================================================================

def _print_results_table(results: List[BenchmarkResult]) -> None:
    """Print a formatted summary table of benchmark results."""
    if not results:
        print("No results to display.")
        return

    # Group by suite
    suites: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        suites.setdefault(r.suite, []).append(r)

    print("\n" + "=" * 100)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    header = (
        f"{'Suite':<14s} {'Name':<40s} {'Batch':<7s} "
        f"{'Throughput':>14s} {'Mean (ms)':>10s} {'Std (ms)':>10s}"
    )
    print(header)
    print("-" * 100)

    for suite_name in [
        "operators", "gradients", "quantifiers", "amp",
        "rules", "e2e", "memory",
    ]:
        if suite_name not in suites:
            continue
        for r in suites[suite_name]:
            throughput_str = (
                f"{r.throughput:>12.0f}" if r.throughput > 0 else "       N/A"
            )
            print(
                f"{r.suite:<14s} {r.name:<40s} {r.batch_size:<7d} "
                f"{throughput_str:>14s} {r.mean_latency_ms:>10.3f} "
                f"{r.std_latency_ms:>10.3f}"
            )
        print("-" * 100)

    print(f"\nTotal benchmarks: {len(results)}")
    print(f"Device: {results[0].device if results else 'N/A'}")


def _save_results_json(results: List[BenchmarkResult], output_path: str) -> None:
    """Save benchmark results to a JSON file."""
    data = {
        "metadata": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device": str(results[0].device) if results else "N/A",
            "num_benchmarks": len(results),
        },
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suite for the Neuro-Symbolic Engine operators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python operator_benchmark.py                        # All benchmarks on CPU
    python operator_benchmark.py --suite operators      # Operator throughput only
    python operator_benchmark.py --suite gradients      # Gradient benchmarks only
    python operator_benchmark.py --suite quantifiers    # Quantifier benchmarks only
    python operator_benchmark.py --suite amp            # AMP benchmarks (needs CUDA)
    python operator_benchmark.py --suite rules          # Rule scoring benchmarks
    python operator_benchmark.py --suite e2e            # End-to-end benchmarks
    python operator_benchmark.py --suite memory         # Memory profiling
    python operator_benchmark.py --device cuda          # GPU benchmarks
    python operator_benchmark.py --output results.json  # Save results to JSON
    python operator_benchmark.py --warmup 20 --iterations 200  # More iterations
        """,
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=[
            "operators", "gradients", "quantifiers",
            "amp", "rules", "e2e", "memory",
        ],
        default=None,
        help="Run a specific benchmark suite. If not specified, runs all suites.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmarks on (default: cpu).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a JSON file at the given path.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before timing (default: 10).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of timed iterations per benchmark (default: 100).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  Neuro-Symbolic Engine -- Operator Benchmark Suite")
    print("=" * 72)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device:     {torch.cuda.get_device_name(0)}")
    print(f"  Device:          {args.device}")
    print(f"  Warmup:          {args.warmup}")
    print(f"  Iterations:      {args.iterations}")
    print("=" * 72)

    benchmark = OperatorBenchmark(
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    if args.suite:
        results = benchmark.run_suite(args.suite)
    else:
        results = benchmark.run_all()

    # Print summary table
    _print_results_table(results)

    # Optionally save to JSON
    if args.output:
        _save_results_json(results, args.output)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
