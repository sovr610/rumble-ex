#!/usr/bin/env python3
"""
run_benchmarks.py -- CLI for running benchmark suites against brain_ai models.

Supports all 7 training phases, dev and production modes, JSON/CSV reporting,
and baseline comparison.  Uses only synthetic data when --mode=dev (no real
datasets required).  Production mode stubs are provided for future integration.

Usage:
    python run_benchmarks.py                            # Run all phases, dev mode
    python run_benchmarks.py --phase 1                  # Phase 1 only
    python run_benchmarks.py --phase 1 3 7              # Specific phases
    python run_benchmarks.py --mode production           # Production benchmarks
    python run_benchmarks.py --output-dir runs/exp1     # Custom output directory
    python run_benchmarks.py --baseline runs/exp0       # Compare against baseline
    python run_benchmarks.py --list                     # List available benchmarks
    python run_benchmarks.py --verbose                  # Verbose progress
    python run_benchmarks.py --batch-size 32            # Override batch size
    python run_benchmarks.py --seed 123                 # Override random seed

Dependencies: torch, numpy (standard for brain_ai)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
import textwrap
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# NumPy 2.0+ renamed trapz -> trapezoid
_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# _utcnow() deprecated in 3.12+; use timezone-aware alternative
try:
    from datetime import timezone as _tz
    def _utcnow() -> datetime:
        return datetime.now(_tz.utc)
except Exception:
    def _utcnow() -> datetime:
        return _utcnow()


# ======================================================================
# Phase / benchmark registry
# ======================================================================

PHASE_INFO: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "SNN Core",
        "task_type": "classify",
        "dev_datasets": ["mnist_synthetic"],
        "prod_datasets": ["cifar10", "cifar100"],
        "num_classes_dev": 10,
        "num_classes_prod": 10,
        "input_shape": (1, 28, 28),
        "input_dim": 784,
    },
    2: {
        "name": "Modality Encoders",
        "task_type": "classify",
        "dev_datasets": ["mnist_multimodal_synthetic"],
        "prod_datasets": ["imagenet1k", "librispeech", "wikitext103"],
        "num_classes_dev": 10,
        "num_classes_prod": 1000,
        "input_shape": (1, 28, 28),
        "input_dim": 784,
    },
    3: {
        "name": "HTM Temporal",
        "task_type": "anomaly",
        "dev_datasets": ["synthetic_sequences"],
        "prod_datasets": ["nab", "yahoo_s5"],
        "seq_len_dev": 200,
        "seq_len_prod": 1000,
    },
    4: {
        "name": "Global Workspace",
        "task_type": "classify",
        "dev_datasets": ["simple_multimodal_synthetic"],
        "prod_datasets": ["vqa_v2", "cmu_multimodal_sdk"],
        "num_classes_dev": 10,
        "num_classes_prod": 3129,
        "input_shape": (1, 28, 28),
        "input_dim": 784,
    },
    5: {
        "name": "Active Inference",
        "task_type": "classify",
        "dev_datasets": ["cartpole_synthetic", "mountaincar_synthetic"],
        "prod_datasets": ["d4rl_halfcheetah", "minari_suite"],
        "num_classes_dev": 2,
        "num_classes_prod": 4,
        "input_shape": (4,),
        "input_dim": 4,
    },
    6: {
        "name": "Reasoning",
        "task_type": "reasoning",
        "dev_datasets": ["mini_babi_synthetic"],
        "prod_datasets": ["babi_full", "proofwriter", "folio"],
        "num_tasks_dev": 20,
        "num_tasks_prod": 1000,
    },
    7: {
        "name": "Meta-Learning",
        "task_type": "few_shot",
        "dev_datasets": ["omniglot_synthetic"],
        "prod_datasets": ["mini_imagenet", "tiered_imagenet"],
        "n_way": 5,
        "k_shot_dev": 1,
        "k_shot_prod": 5,
        "feat_dim": 64,
    },
}


# ======================================================================
# Inline helpers (standalone, no cross-file imports)
# ======================================================================

@dataclass
class _BenchConfig:
    """Minimal inline configuration for standalone operation."""
    task_type: str = "classify"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro"])
    num_classes: int = 10
    batch_size: int = 64
    device: str = "cpu"
    seed: int = 42
    n_way: int = 5
    k_shot: int = 1
    n_query: int = 15
    n_episodes: int = 100
    anomaly_window: int = 100
    n_thresholds: int = 50
    nab_profile: str = "standard"
    reasoning_task: str = "babi"
    save_confusion_matrix: bool = True
    save_per_class: bool = True
    baseline_path: Optional[str] = None
    report_format: str = "json"
    output_dir: str = "runs"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _MetricsSuite:
    """Inline metrics suite for standalone benchmark runs."""

    def __init__(self, num_classes: int = 10, device: str = "cpu") -> None:
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._cm = torch.zeros(self.num_classes, self.num_classes,
                               dtype=torch.int64, device=self.device)
        self._total = 0

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.numel() == 0:
            return
        targets = targets.to(self.device).long()
        predictions = predictions.to(self.device)
        pred_labels = predictions.long() if predictions.dim() == 1 else predictions.argmax(dim=-1)
        for t, p in zip(targets, pred_labels):
            ti, pi = t.item(), p.item()
            if 0 <= ti < self.num_classes and 0 <= pi < self.num_classes:
                self._cm[ti, pi] += 1
        self._total += targets.shape[0]

    def compute(self) -> Dict[str, float]:
        if self._total == 0:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}
        tp = self._cm.diag()
        fp = self._cm.sum(dim=0) - tp
        fn = self._cm.sum(dim=1) - tp
        support = self._cm.sum(dim=1)
        acc = float(tp.sum()) / float(self._total)
        prec = self._sdiv(tp.float(), (tp + fp).float())
        rec = self._sdiv(tp.float(), (tp + fn).float())
        f1 = self._sdiv(2.0 * prec * rec, prec + rec)
        f1_macro = float(f1.mean())
        total_s = support.sum().float()
        f1_w = float((f1 * support.float() / total_s.clamp(min=1)).sum()) if total_s > 0 else 0.0
        return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_w}

    def per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        tp = self._cm.diag()
        fp = self._cm.sum(dim=0) - tp
        fn = self._cm.sum(dim=1) - tp
        support = self._cm.sum(dim=1)
        prec = self._sdiv(tp.float(), (tp + fp).float())
        rec = self._sdiv(tp.float(), (tp + fn).float())
        f1 = self._sdiv(2.0 * prec * rec, prec + rec)
        out: Dict[int, Dict[str, float]] = {}
        for c in range(self.num_classes):
            out[c] = {
                "precision": float(prec[c]), "recall": float(rec[c]),
                "f1": float(f1[c]), "support": int(support[c]),
            }
        return out

    def confusion_matrix(self) -> Tensor:
        return self._cm.float()

    @staticmethod
    def _sdiv(num: Tensor, den: Tensor) -> Tensor:
        out = torch.zeros_like(num)
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out


class _MockModel(nn.Module):
    """Mock model for dev-mode benchmarking."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: Dict[str, Tensor], **kw) -> Tensor:
        x = list(inputs.values())[0]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.shape[-1] > self.fc.in_features:
            x = x[:, :self.fc.in_features]
        elif x.shape[-1] < self.fc.in_features:
            pad = torch.zeros(x.shape[0], self.fc.in_features - x.shape[-1],
                              device=x.device)
            x = torch.cat([x, pad], dim=-1)
        return self.fc(x)


# ======================================================================
# Synthetic data generators
# ======================================================================

def _generate_classification_data(
    n_samples: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """Generate synthetic classification data."""
    torch.manual_seed(seed)
    data = torch.randn(n_samples, *input_shape)
    labels = torch.randint(0, num_classes, (n_samples,))
    return data, labels


def _generate_anomaly_data(
    n_sequences: int,
    seq_len: int,
    anomaly_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic anomaly detection data (sine + noise + injected anomalies)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 20 * np.pi, seq_len)
    scores_all = []
    labels_all = []
    for _ in range(n_sequences):
        signal = np.sin(t) + rng.randn(seq_len) * 0.1
        labels = np.zeros(seq_len, dtype=int)
        n_anomalies = max(1, int(seq_len * anomaly_fraction))
        positions = rng.choice(seq_len, n_anomalies, replace=False)
        signal[positions] += rng.uniform(3.0, 8.0, size=n_anomalies)
        labels[positions] = 1
        scores_all.append(np.abs(signal))
        labels_all.append(labels)
    return np.concatenate(scores_all), np.concatenate(labels_all)


def _generate_reasoning_tasks(
    n_tasks: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate synthetic bAbI-style reasoning tasks."""
    import string
    rng = np.random.RandomState(seed)
    entities = ["bathroom", "kitchen", "garden", "bedroom", "office",
                "hallway", "basement", "attic", "garage", "study"]
    names = ["John", "Mary", "Sandra", "Daniel", "Fred",
             "Alice", "Bob", "Eve", "Trent", "Carol"]
    tasks = []
    for _ in range(n_tasks):
        name = names[rng.randint(len(names))]
        loc1 = entities[rng.randint(len(entities))]
        loc2 = entities[rng.randint(len(entities))]
        context = f"{name} went to the {loc1}. {name} moved to the {loc2}."
        question = f"Where is {name}?"
        answer = loc2
        # Simulate a prediction (correct 70% of the time for realistic benchmark)
        if rng.random() < 0.7:
            predicted = answer
        else:
            predicted = entities[rng.randint(len(entities))]
        tasks.append({
            "context": context,
            "question": question,
            "answer": answer,
            "predicted": predicted,
        })
    return tasks


def _generate_few_shot_data(
    n_classes: int,
    samples_per_class: int,
    feat_dim: int,
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """Generate clustered feature data for few-shot benchmarking."""
    torch.manual_seed(seed)
    features = []
    labels = []
    for c in range(n_classes):
        center = torch.randn(feat_dim) * 3
        feats = center.unsqueeze(0) + torch.randn(samples_per_class, feat_dim) * 0.5
        features.append(feats)
        labels.extend([c] * samples_per_class)
    return torch.cat(features, dim=0), torch.tensor(labels, dtype=torch.long)


# ======================================================================
# Benchmark result dataclass
# ======================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    phase: int
    dataset: str
    task_type: str
    metrics: Dict[str, float]
    duration_seconds: float
    num_samples: int
    per_class: Dict[int, Dict[str, float]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    report_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = _utcnow().isoformat()
        if not self.report_id:
            short = uuid.uuid4().hex[:6]
            ts = _utcnow().strftime("%Y%m%d_%H%M%S")
            self.report_id = f"bench_p{self.phase}_{ts}_{short}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "metadata": {
                "report_id": self.report_id,
                "timestamp": self.timestamp,
                "phase": self.phase,
                "dataset": self.dataset,
                "task_type": self.task_type,
                "num_samples": self.num_samples,
            },
            "per_metric": _nan_to_none(self.metrics),
            "per_class": {
                str(k): _nan_to_none(v) for k, v in self.per_class.items()
            },
            "timing": {
                "total_seconds": self.duration_seconds,
                "per_sample_ms": (
                    self.duration_seconds * 1000.0 / max(self.num_samples, 1)
                ),
            },
            "extra": self.extra,
        }


def _nan_to_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Replace NaN values with None for JSON serialisation."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


# ======================================================================
# Phase-specific benchmark runners
# ======================================================================

def _run_classification_benchmark(
    phase: int,
    info: Dict[str, Any],
    config: _BenchConfig,
    mode: str,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a classification benchmark for phases 1, 2, 4, 5."""
    nc = info.get("num_classes_dev" if mode == "dev" else "num_classes_prod", 10)
    input_dim = info.get("input_dim", 784)
    input_shape = info.get("input_shape", (1, 28, 28))
    n_samples = 256 if mode == "dev" else 1024
    datasets = info["dev_datasets"] if mode == "dev" else info["prod_datasets"]
    ds_name = datasets[0]

    if verbose:
        print(f"  Generating {n_samples} synthetic samples "
              f"(shape={input_shape}, classes={nc})")

    data, labels = _generate_classification_data(
        n_samples, input_shape, nc, seed=config.seed
    )

    model = _MockModel(input_dim=input_dim, num_classes=nc)
    loader = DataLoader(
        TensorDataset(data, labels),
        batch_size=config.batch_size,
        shuffle=False,
    )

    suite = _MetricsSuite(num_classes=nc, device=config.device)
    suite.reset()
    model.train(False)

    t0 = time.time()
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            out = model({"input": batch_data})
            suite.update(out, batch_labels)
    duration = time.time() - t0

    metrics = suite.compute()
    per_class = suite.per_class_metrics() if config.save_per_class else {}

    if verbose:
        print(f"  Results: accuracy={metrics['accuracy']:.4f}, "
              f"f1_macro={metrics['f1_macro']:.4f}, "
              f"duration={duration:.3f}s")

    return BenchmarkResult(
        phase=phase,
        dataset=ds_name,
        task_type="classify",
        metrics=metrics,
        duration_seconds=duration,
        num_samples=n_samples,
        per_class=per_class,
    )


def _run_anomaly_benchmark(
    phase: int,
    info: Dict[str, Any],
    config: _BenchConfig,
    mode: str,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run an anomaly detection benchmark for phase 3."""
    seq_len = info.get("seq_len_dev" if mode == "dev" else "seq_len_prod", 200)
    n_sequences = 16 if mode == "dev" else 64
    datasets = info["dev_datasets"] if mode == "dev" else info["prod_datasets"]
    ds_name = datasets[0]

    if verbose:
        print(f"  Generating {n_sequences} sequences (len={seq_len})")

    t0 = time.time()
    scores, labels = _generate_anomaly_data(
        n_sequences, seq_len, anomaly_fraction=0.05, seed=config.seed
    )

    # Threshold sweep
    thresholds = np.linspace(
        float(scores.min()), float(scores.max()), config.n_thresholds
    )
    best_f1 = -1.0
    best_threshold = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2.0 * p * r / max(p + r, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
            best_prec = p
            best_rec = r

    # Compute AUROC
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(-scores)
        sorted_labels = labels[order]
        tp_cum = np.cumsum(sorted_labels)
        fp_cum = np.cumsum(1 - sorted_labels)
        tpr = np.concatenate([[0], tp_cum / n_pos])
        fpr = np.concatenate([[0], fp_cum / n_neg])
        auroc = float(_np_trapz(tpr, fpr))
    else:
        auroc = float("nan")

    duration = time.time() - t0
    n_total = n_sequences * seq_len

    metrics = {
        "precision": best_prec,
        "recall": best_rec,
        "f1": best_f1,
        "auroc": auroc,
        "best_threshold": best_threshold,
    }

    if verbose:
        auroc_str = f"{auroc:.4f}" if not math.isnan(auroc) else "NaN"
        print(f"  Results: f1={best_f1:.4f}, auroc={auroc_str}, "
              f"threshold={best_threshold:.3f}, duration={duration:.3f}s")

    return BenchmarkResult(
        phase=phase,
        dataset=ds_name,
        task_type="anomaly",
        metrics=metrics,
        duration_seconds=duration,
        num_samples=n_total,
        extra={"n_sequences": n_sequences, "seq_len": seq_len,
               "n_positives": n_pos, "n_negatives": n_neg},
    )


def _run_reasoning_benchmark(
    phase: int,
    info: Dict[str, Any],
    config: _BenchConfig,
    mode: str,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a reasoning benchmark for phase 6."""
    n_tasks = info.get("num_tasks_dev" if mode == "dev" else "num_tasks_prod", 20)
    datasets = info["dev_datasets"] if mode == "dev" else info["prod_datasets"]
    ds_name = datasets[0]

    if verbose:
        print(f"  Generating {n_tasks} reasoning tasks")

    t0 = time.time()
    tasks = _generate_reasoning_tasks(n_tasks, seed=config.seed)

    # Compute exact match
    import string
    stop_words = {"a", "an", "the"}

    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = [t for t in text.split() if t not in stop_words]
        return " ".join(tokens).strip()

    correct = 0
    for task in tasks:
        if normalize(task["predicted"]) == normalize(task["answer"]):
            correct += 1
    exact_match = correct / max(len(tasks), 1)

    # Logical consistency (simplified: check that answers are single-word entities)
    consistent = sum(
        1 for task in tasks
        if len(normalize(task["predicted"]).split()) <= 2
    )
    consistency_rate = consistent / max(len(tasks), 1)

    duration = time.time() - t0

    metrics = {
        "exact_match": exact_match,
        "consistency_rate": consistency_rate,
        "n_correct": float(correct),
        "n_total": float(len(tasks)),
    }

    if verbose:
        print(f"  Results: exact_match={exact_match:.4f}, "
              f"consistency={consistency_rate:.4f}, "
              f"duration={duration:.3f}s")

    return BenchmarkResult(
        phase=phase,
        dataset=ds_name,
        task_type="reasoning",
        metrics=metrics,
        duration_seconds=duration,
        num_samples=n_tasks,
    )


def _run_few_shot_benchmark(
    phase: int,
    info: Dict[str, Any],
    config: _BenchConfig,
    mode: str,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a few-shot benchmark for phase 7."""
    n_way = info.get("n_way", 5)
    k_shot = info.get("k_shot_dev" if mode == "dev" else "k_shot_prod", 1)
    feat_dim = info.get("feat_dim", 64)
    n_total_classes = n_way * 4  # pool of classes
    spc = 30  # samples per class
    n_episodes = config.n_episodes
    n_query = config.n_query
    datasets = info["dev_datasets"] if mode == "dev" else info["prod_datasets"]
    ds_name = datasets[0]

    if verbose:
        print(f"  Running {n_episodes} episodes ({n_way}-way {k_shot}-shot)")

    t0 = time.time()
    features, labels = _generate_few_shot_data(
        n_total_classes, spc, feat_dim, seed=config.seed
    )
    rng = np.random.RandomState(config.seed)

    episode_accs = []
    n_collapsed = 0

    for ep in range(n_episodes):
        # Sample N classes
        chosen_classes = rng.choice(n_total_classes, n_way, replace=False)

        support_feats = []
        support_labels = []
        query_feats = []
        query_labels = []

        for new_label, c in enumerate(chosen_classes):
            mask = (labels == c).nonzero(as_tuple=True)[0].tolist()
            if len(mask) < k_shot + n_query:
                selected = rng.choice(mask, k_shot + n_query, replace=True)
            else:
                selected = rng.choice(mask, k_shot + n_query, replace=False)
            s_idx = selected[:k_shot]
            q_idx = selected[k_shot:]
            support_feats.append(features[s_idx])
            support_labels.extend([new_label] * k_shot)
            query_feats.append(features[q_idx])
            query_labels.extend([new_label] * n_query)

        # Prototypical classification
        support_feats_t = torch.cat(support_feats, dim=0)
        query_feats_t = torch.cat(query_feats, dim=0)
        query_labels_t = torch.tensor(query_labels, dtype=torch.long)

        # Compute prototypes (mean of support per class)
        prototypes = torch.zeros(n_way, feat_dim)
        for cls_i in range(n_way):
            cls_mask = torch.tensor(support_labels) == cls_i
            prototypes[cls_i] = support_feats_t[cls_mask].mean(dim=0)

        # Classify queries by nearest prototype
        dists = torch.cdist(query_feats_t, prototypes)
        pred_labels = dists.argmin(dim=1)
        acc = float((pred_labels == query_labels_t).float().mean())
        episode_accs.append(acc)

        # Check for collapse (all predictions same class)
        if pred_labels.unique().numel() == 1:
            n_collapsed += 1

    episode_accs_np = np.array(episode_accs)
    mean_acc = float(episode_accs_np.mean())
    std_acc = float(episode_accs_np.std())
    ci_95 = 1.96 * std_acc / math.sqrt(max(n_episodes, 1))
    is_collapsed = n_collapsed > n_episodes // 2

    duration = time.time() - t0

    metrics = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci_95": ci_95,
        "accuracy_lower": mean_acc - ci_95,
        "accuracy_upper": mean_acc + ci_95,
    }

    if verbose:
        print(f"  Results: mean_acc={mean_acc:.4f} +/- {ci_95:.4f}, "
              f"collapsed={n_collapsed}/{n_episodes}, "
              f"duration={duration:.3f}s")

    return BenchmarkResult(
        phase=phase,
        dataset=ds_name,
        task_type="few_shot",
        metrics=metrics,
        duration_seconds=duration,
        num_samples=n_episodes * n_way * n_query,
        extra={"n_way": n_way, "k_shot": k_shot, "n_episodes": n_episodes,
               "n_collapsed": n_collapsed, "is_collapsed": is_collapsed},
    )


# ======================================================================
# Dispatch table
# ======================================================================

_RUNNERS: Dict[str, Callable] = {
    "classify": _run_classification_benchmark,
    "anomaly": _run_anomaly_benchmark,
    "reasoning": _run_reasoning_benchmark,
    "few_shot": _run_few_shot_benchmark,
}


def run_phase(
    phase: int,
    config: _BenchConfig,
    mode: str = "dev",
    verbose: bool = False,
) -> BenchmarkResult:
    """Run the benchmark for a single phase."""
    if phase not in PHASE_INFO:
        raise ValueError(f"Unknown phase {phase}. Valid: 1-7")
    info = PHASE_INFO[phase]
    task_type = info["task_type"]
    runner = _RUNNERS.get(task_type)
    if runner is None:
        raise ValueError(f"No runner for task_type={task_type} (phase {phase})")
    return runner(phase, info, config, mode, verbose)


# ======================================================================
# Report writing
# ======================================================================

def save_json_report(result: BenchmarkResult, output_dir: str) -> str:
    """Save a benchmark result as a JSON report. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result.report_id}.json")
    report = result.to_dict()
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path


def save_csv_summary(results: List[BenchmarkResult], output_dir: str) -> str:
    """Save a CSV summary of multiple benchmark results. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = _utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_summary_{ts}.csv")

    if not results:
        with open(path, "w") as fh:
            fh.write("phase,dataset,task_type\n")
        return path

    # Collect all metric keys
    all_metric_keys: List[str] = []
    seen = set()
    for r in results:
        for k in r.metrics:
            if k not in seen:
                all_metric_keys.append(k)
                seen.add(k)

    fieldnames = ["phase", "dataset", "task_type", "num_samples",
                  "duration_seconds"] + all_metric_keys

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: Dict[str, Any] = {
                "phase": r.phase,
                "dataset": r.dataset,
                "task_type": r.task_type,
                "num_samples": r.num_samples,
                "duration_seconds": f"{r.duration_seconds:.4f}",
            }
            for k in all_metric_keys:
                v = r.metrics.get(k, "")
                if isinstance(v, float):
                    if math.isnan(v):
                        row[k] = ""
                    else:
                        row[k] = f"{v:.6f}"
                else:
                    row[k] = v
            writer.writerow(row)
    return path


# ======================================================================
# Comparison engine
# ======================================================================

HIGHER_IS_BETTER = {
    "accuracy", "f1_macro", "f1_weighted", "auroc", "precision", "recall",
    "f1", "mean_accuracy", "exact_match", "consistency_rate",
}

LOWER_IS_BETTER = {
    "loss", "forgetting", "best_threshold",
}


def compare_with_baseline(
    results: List[BenchmarkResult],
    baseline_dir: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare current results against baseline JSON reports."""
    comparison: Dict[str, Any] = {"phases": {}, "summary": {}}
    improved = 0
    degraded = 0
    unchanged = 0

    for result in results:
        phase_key = f"phase_{result.phase}"

        # Look for matching baseline
        baseline_report = None
        if os.path.isdir(baseline_dir):
            for fname in os.listdir(baseline_dir):
                if fname.endswith(".json") and f"_p{result.phase}_" in fname:
                    fpath = os.path.join(baseline_dir, fname)
                    try:
                        with open(fpath) as fh:
                            baseline_report = json.load(fh)
                    except (json.JSONDecodeError, IOError):
                        pass
                    break

        if baseline_report is None:
            comparison["phases"][phase_key] = {
                "status": "no_baseline",
                "current": _nan_to_none(result.metrics),
            }
            continue

        base_metrics = baseline_report.get("per_metric", {})
        deltas: Dict[str, Dict[str, Any]] = {}

        for metric_name, current_val in result.metrics.items():
            if isinstance(current_val, float) and math.isnan(current_val):
                continue
            base_val = base_metrics.get(metric_name)
            if base_val is None:
                continue

            delta = current_val - base_val
            abs_delta = abs(delta)
            if abs_delta < 1e-4:
                status = "unchanged"
                unchanged += 1
            elif metric_name in HIGHER_IS_BETTER:
                status = "improved" if delta > 0 else "degraded"
            elif metric_name in LOWER_IS_BETTER:
                status = "improved" if delta < 0 else "degraded"
            else:
                status = "changed"

            if status == "improved":
                improved += 1
            elif status == "degraded":
                degraded += 1

            deltas[metric_name] = {
                "current": current_val,
                "baseline": base_val,
                "delta": delta,
                "relative": delta / abs(base_val) if abs(base_val) > 1e-8 else None,
                "status": status,
            }

        comparison["phases"][phase_key] = {
            "status": "compared",
            "deltas": deltas,
        }

        if verbose and deltas:
            print(f"\n  Phase {result.phase} comparison:")
            for mn, md in deltas.items():
                sign = "+" if md["delta"] > 0 else ""
                print(f"    {mn}: {md['baseline']:.4f} -> {md['current']:.4f} "
                      f"({sign}{md['delta']:.4f}) [{md['status']}]")

    comparison["summary"] = {
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "total_compared": improved + degraded + unchanged,
    }
    return comparison


# ======================================================================
# Main CLI
# ======================================================================

def list_benchmarks() -> None:
    """Print all available benchmarks."""
    print("Available benchmarks:")
    print("-" * 72)
    for phase, info in sorted(PHASE_INFO.items()):
        print(f"\n  Phase {phase}: {info['name']}")
        print(f"    Task type: {info['task_type']}")
        print(f"    Dev datasets: {', '.join(info['dev_datasets'])}")
        print(f"    Prod datasets: {', '.join(info['prod_datasets'])}")
    print(f"\nTotal: {len(PHASE_INFO)} phases")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark suites for the brain_ai assessment infrastructure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_benchmarks.py                            # All phases, dev
              python run_benchmarks.py --phase 1 3 7              # Specific phases
              python run_benchmarks.py --mode production          # Production mode
              python run_benchmarks.py --output-dir runs/exp1     # Custom output
              python run_benchmarks.py --baseline runs/exp0       # Compare baseline
        """),
    )
    parser.add_argument(
        "--phase", type=int, nargs="+", default=None,
        help="Phase(s) to benchmark (1-7). Default: all phases.",
    )
    parser.add_argument(
        "--mode", type=str, default="dev", choices=["dev", "production"],
        help="Benchmark mode: dev (synthetic data) or production. Default: dev.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/benchmarks",
        help="Directory for output reports. Default: runs/benchmarks.",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline report directory for comparison.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Override batch size. Default: 64.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of few-shot episodes. Default: 100.",
    )
    parser.add_argument(
        "--format", type=str, default="json", choices=["json", "csv", "both"],
        help="Report format. Default: json.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available benchmarks and exit.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run benchmarks but do not save reports to disk.",
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    # Determine phases
    phases = args.phase if args.phase else list(range(1, 8))
    for p in phases:
        if p not in PHASE_INFO:
            print(f"ERROR: Unknown phase {p}. Valid phases: 1-7.", file=sys.stderr)
            sys.exit(1)

    # Build config
    config = _BenchConfig(
        batch_size=args.batch_size,
        seed=args.seed,
        n_episodes=args.n_episodes,
        report_format=args.format,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("Brain-AI Benchmark Runner")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Phases:     {phases}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Seed:       {config.seed}")
    print(f"  Output:     {args.output_dir}")
    if args.baseline:
        print(f"  Baseline:   {args.baseline}")
    print("=" * 60)

    # Run benchmarks
    results: List[BenchmarkResult] = []
    total_t0 = time.time()

    for phase in phases:
        info = PHASE_INFO[phase]
        print(f"\n--- Phase {phase}: {info['name']} ({info['task_type']}) ---")
        try:
            result = run_phase(phase, config, mode=args.mode, verbose=args.verbose)
            results.append(result)

            # Print brief summary
            if not args.verbose:
                primary_metric = _get_primary_metric(result)
                if primary_metric:
                    mk, mv = primary_metric
                    mv_str = f"{mv:.4f}" if not (isinstance(mv, float) and math.isnan(mv)) else "NaN"
                    print(f"  {mk}={mv_str}  ({result.duration_seconds:.2f}s)")
                else:
                    print(f"  Completed ({result.duration_seconds:.2f}s)")

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()

    total_duration = time.time() - total_t0

    # Save reports
    saved_paths: List[str] = []
    if not args.no_save and results:
        if args.format in ("json", "both"):
            for r in results:
                path = save_json_report(r, args.output_dir)
                saved_paths.append(path)
                if args.verbose:
                    print(f"  Saved: {path}")

        if args.format in ("csv", "both"):
            path = save_csv_summary(results, args.output_dir)
            saved_paths.append(path)
            if args.verbose:
                print(f"  Saved: {path}")

    # Baseline comparison
    comparison = None
    if args.baseline and results:
        print(f"\n--- Baseline Comparison ---")
        comparison = compare_with_baseline(results, args.baseline, verbose=args.verbose)
        summary = comparison["summary"]
        print(f"  Improved:  {summary['improved']}")
        print(f"  Degraded:  {summary['degraded']}")
        print(f"  Unchanged: {summary['unchanged']}")

        # Save comparison report
        if not args.no_save:
            comp_path = os.path.join(args.output_dir, "comparison_report.json")
            with open(comp_path, "w") as fh:
                json.dump(comparison, fh, indent=2, default=str)
            saved_paths.append(comp_path)

    # Final summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Phases run: {len(results)}/{len(phases)}")
    print(f"  Total time: {total_duration:.2f}s")

    if results:
        for r in results:
            primary = _get_primary_metric(r)
            if primary:
                mk, mv = primary
                mv_str = f"{mv:.4f}" if not (isinstance(mv, float) and math.isnan(mv)) else "NaN"
                print(f"  Phase {r.phase} ({r.task_type}): {mk}={mv_str}")
            else:
                print(f"  Phase {r.phase} ({r.task_type}): completed")

    if saved_paths:
        print(f"\n  Reports saved to: {args.output_dir}/")
        for p in saved_paths:
            print(f"    {os.path.basename(p)}")

    if len(results) < len(phases):
        print("\n  WARNING: Some phases failed to complete.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("ALL BENCHMARKS COMPLETED")
    print(f"{'=' * 60}")


def _get_primary_metric(result: BenchmarkResult) -> Optional[Tuple[str, float]]:
    """Get the primary metric for display."""
    priority = ["accuracy", "mean_accuracy", "exact_match", "f1", "auroc"]
    for key in priority:
        if key in result.metrics:
            return key, result.metrics[key]
    # Fall back to first metric
    if result.metrics:
        k = next(iter(result.metrics))
        return k, result.metrics[k]
    return None


# ======================================================================
# Self-tests
# ======================================================================

def _run_self_tests() -> None:
    """Run self-tests to verify this script works correctly."""
    passed = 0
    failed = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    def approx(a: float, b: float, tol: float = 1e-4) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) < tol

    print("Running self-tests...\n")

    # --- Phase registry ---
    check("registry_has_7_phases", len(PHASE_INFO) == 7)
    check("phase_keys_1_to_7", set(PHASE_INFO.keys()) == {1, 2, 3, 4, 5, 6, 7})

    for p in range(1, 8):
        info = PHASE_INFO[p]
        check(f"phase_{p}_has_name", "name" in info)
        check(f"phase_{p}_has_task_type", "task_type" in info)
        check(f"phase_{p}_has_dev_datasets", "dev_datasets" in info)
        check(f"phase_{p}_has_prod_datasets", "prod_datasets" in info)
        check(f"phase_{p}_task_type_known", info["task_type"] in _RUNNERS)

    # --- BenchConfig ---
    cfg = _BenchConfig()
    check("config_default_task_type", cfg.task_type == "classify")
    check("config_default_batch_size", cfg.batch_size == 64)
    check("config_default_seed", cfg.seed == 42)
    check("config_to_dict", isinstance(cfg.to_dict(), dict))
    check("config_to_dict_has_task_type", "task_type" in cfg.to_dict())

    # --- MetricsSuite ---
    suite = _MetricsSuite(3)
    suite.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    m = suite.compute()
    check("suite_accuracy_perfect", approx(m["accuracy"], 1.0))
    check("suite_f1_perfect", approx(m["f1_macro"], 1.0))

    suite2 = _MetricsSuite(3)
    suite2.update(torch.tensor([1, 2, 0]), torch.tensor([0, 1, 2]))
    m2 = suite2.compute()
    check("suite_accuracy_zero", approx(m2["accuracy"], 0.0))

    suite3 = _MetricsSuite(3)
    m3 = suite3.compute()
    check("suite_empty_accuracy", approx(m3["accuracy"], 0.0))

    suite4 = _MetricsSuite(2)
    suite4.update(torch.tensor([0, 0, 0, 0, 1]), torch.tensor([0, 0, 0, 1, 1]))
    pc = suite4.per_class_metrics()
    check("per_class_keys", 0 in pc and 1 in pc)
    check("per_class_prec_0", approx(pc[0]["precision"], 0.75))
    check("per_class_rec_0", approx(pc[0]["recall"], 1.0))
    check("per_class_prec_1", approx(pc[1]["precision"], 1.0))
    check("per_class_rec_1", approx(pc[1]["recall"], 0.5))

    cm = suite4.confusion_matrix()
    check("cm_shape", cm.shape == (2, 2))

    # --- MockModel ---
    model = _MockModel(input_dim=8, num_classes=3)
    out = model({"x": torch.randn(4, 8)})
    check("mock_model_output_shape", out.shape == (4, 3))

    model2 = _MockModel(input_dim=784, num_classes=10)
    out2 = model2({"vision": torch.randn(2, 1, 28, 28)})
    check("mock_model_flatten", out2.shape == (2, 10))

    # --- Synthetic data generators ---
    data, labels = _generate_classification_data(100, (1, 28, 28), 10, seed=0)
    check("gen_class_data_shape", data.shape == (100, 1, 28, 28))
    check("gen_class_labels_shape", labels.shape == (100,))
    check("gen_class_labels_range", labels.min() >= 0 and labels.max() < 10)

    scores, alabs = _generate_anomaly_data(4, 200, anomaly_fraction=0.05, seed=0)
    check("gen_anomaly_scores_len", len(scores) == 800)
    check("gen_anomaly_labels_len", len(alabs) == 800)
    check("gen_anomaly_has_positives", alabs.sum() > 0)
    check("gen_anomaly_has_negatives", (alabs == 0).sum() > 0)

    tasks = _generate_reasoning_tasks(50, seed=0)
    check("gen_reasoning_count", len(tasks) == 50)
    check("gen_reasoning_has_answer", all("answer" in t for t in tasks))
    check("gen_reasoning_has_predicted", all("predicted" in t for t in tasks))

    feats, flabs = _generate_few_shot_data(10, 30, 32, seed=0)
    check("gen_few_shot_feats_shape", feats.shape == (300, 32))
    check("gen_few_shot_labels_shape", flabs.shape == (300,))
    check("gen_few_shot_labels_range", flabs.min() >= 0 and flabs.max() < 10)

    # Determinism
    d1, l1 = _generate_classification_data(10, (4,), 3, seed=42)
    d2, l2 = _generate_classification_data(10, (4,), 3, seed=42)
    check("gen_class_deterministic", torch.equal(d1, d2) and torch.equal(l1, l2))

    # --- BenchmarkResult ---
    br = BenchmarkResult(
        phase=1, dataset="test_ds", task_type="classify",
        metrics={"accuracy": 0.95}, duration_seconds=1.0, num_samples=100,
    )
    check("result_has_report_id", len(br.report_id) > 0)
    check("result_has_timestamp", len(br.timestamp) > 0)
    rd = br.to_dict()
    check("result_to_dict_has_metadata", "metadata" in rd)
    check("result_to_dict_has_per_metric", "per_metric" in rd)
    check("result_to_dict_has_timing", "timing" in rd)
    check("result_metadata_phase", rd["metadata"]["phase"] == 1)

    # NaN handling
    br_nan = BenchmarkResult(
        phase=1, dataset="x", task_type="classify",
        metrics={"accuracy": 0.5, "auroc": float("nan")},
        duration_seconds=0.1, num_samples=10,
    )
    rd_nan = br_nan.to_dict()
    check("result_nan_to_none", rd_nan["per_metric"]["auroc"] is None)
    check("result_nan_keeps_value", rd_nan["per_metric"]["accuracy"] == 0.5)

    # --- Phase runners (dev mode) ---
    cfg_run = _BenchConfig(batch_size=16, seed=42, n_episodes=10)

    for phase in range(1, 8):
        try:
            result = run_phase(phase, cfg_run, mode="dev", verbose=False)
            check(f"run_phase_{phase}_completes", True)
            check(f"run_phase_{phase}_has_metrics", len(result.metrics) > 0)
            check(f"run_phase_{phase}_positive_duration", result.duration_seconds > 0)
            check(f"run_phase_{phase}_has_samples", result.num_samples > 0)
        except Exception as e:
            check(f"run_phase_{phase}_completes", False)
            print(f"    Error: {e}")

    # --- Specific phase result checks ---
    r1 = run_phase(1, cfg_run, mode="dev")
    check("phase1_has_accuracy", "accuracy" in r1.metrics)
    check("phase1_accuracy_range", 0.0 <= r1.metrics["accuracy"] <= 1.0)
    check("phase1_has_f1", "f1_macro" in r1.metrics)

    r3 = run_phase(3, cfg_run, mode="dev")
    check("phase3_has_f1", "f1" in r3.metrics)
    check("phase3_has_auroc", "auroc" in r3.metrics)
    check("phase3_has_threshold", "best_threshold" in r3.metrics)

    r6 = run_phase(6, cfg_run, mode="dev")
    check("phase6_has_exact_match", "exact_match" in r6.metrics)
    check("phase6_exact_match_range", 0.0 <= r6.metrics["exact_match"] <= 1.0)

    r7 = run_phase(7, cfg_run, mode="dev")
    check("phase7_has_mean_accuracy", "mean_accuracy" in r7.metrics)
    check("phase7_has_ci", "ci_95" in r7.metrics)
    check("phase7_ci_positive", r7.metrics["ci_95"] >= 0)

    # --- Report saving ---
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_json_report(r1, tmpdir)
        check("json_report_saved", os.path.exists(path))
        with open(path) as fh:
            loaded = json.load(fh)
        check("json_report_has_metadata", "metadata" in loaded)
        check("json_report_has_per_metric", "per_metric" in loaded)
        check("json_report_round_trip", loaded["per_metric"]["accuracy"] == r1.metrics["accuracy"])

    # --- CSV summary ---
    with tempfile.TemporaryDirectory() as tmpdir:
        all_results = [r1, r3, r6, r7]
        csv_path = save_csv_summary(all_results, tmpdir)
        check("csv_report_saved", os.path.exists(csv_path))
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        check("csv_has_rows", len(rows) == 4)
        check("csv_has_phase_col", "phase" in rows[0])
        check("csv_has_dataset_col", "dataset" in rows[0])

    # --- Empty CSV ---
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path2 = save_csv_summary([], tmpdir)
        check("csv_empty_saved", os.path.exists(csv_path2))

    # --- Comparison engine ---
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save baseline
        baseline_result = BenchmarkResult(
            phase=1, dataset="test", task_type="classify",
            metrics={"accuracy": 0.80, "f1_macro": 0.78},
            duration_seconds=1.0, num_samples=100,
        )
        save_json_report(baseline_result, tmpdir)

        # Current result
        current = BenchmarkResult(
            phase=1, dataset="test", task_type="classify",
            metrics={"accuracy": 0.85, "f1_macro": 0.82},
            duration_seconds=1.0, num_samples=100,
        )
        comp = compare_with_baseline([current], tmpdir)
        check("comparison_has_phases", "phases" in comp)
        check("comparison_has_summary", "summary" in comp)

        p1_comp = comp["phases"].get("phase_1", {})
        if p1_comp.get("status") == "compared":
            deltas = p1_comp.get("deltas", {})
            if "accuracy" in deltas:
                check("comparison_accuracy_improved",
                      deltas["accuracy"]["status"] == "improved")
                check("comparison_accuracy_delta",
                      approx(deltas["accuracy"]["delta"], 0.05))
            else:
                check("comparison_accuracy_found", False)
        else:
            check("comparison_status_compared", False)

    # --- Comparison with no baseline ---
    with tempfile.TemporaryDirectory() as tmpdir:
        comp2 = compare_with_baseline([r1], tmpdir)
        p1_status = comp2["phases"].get("phase_1", {}).get("status")
        check("comparison_no_baseline", p1_status == "no_baseline")

    # --- Comparison degraded metric ---
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_high = BenchmarkResult(
            phase=2, dataset="test", task_type="classify",
            metrics={"accuracy": 0.95},
            duration_seconds=1.0, num_samples=100,
        )
        save_json_report(baseline_high, tmpdir)

        current_low = BenchmarkResult(
            phase=2, dataset="test", task_type="classify",
            metrics={"accuracy": 0.80},
            duration_seconds=1.0, num_samples=100,
        )
        comp3 = compare_with_baseline([current_low], tmpdir)
        p2_comp = comp3["phases"].get("phase_2", {})
        if p2_comp.get("status") == "compared":
            deltas3 = p2_comp.get("deltas", {})
            if "accuracy" in deltas3:
                check("comparison_accuracy_degraded",
                      deltas3["accuracy"]["status"] == "degraded")
            else:
                check("comparison_degraded_accuracy_found", False)
        else:
            check("comparison_degraded_status", False)

    # --- _nan_to_none ---
    cleaned = _nan_to_none({"a": 1.0, "b": float("nan"), "c": "text"})
    check("nan_to_none_keeps_value", cleaned["a"] == 1.0)
    check("nan_to_none_converts_nan", cleaned["b"] is None)
    check("nan_to_none_keeps_string", cleaned["c"] == "text")

    # --- Higher/lower is better sets ---
    check("higher_has_accuracy", "accuracy" in HIGHER_IS_BETTER)
    check("higher_has_auroc", "auroc" in HIGHER_IS_BETTER)
    check("lower_has_loss", "loss" in LOWER_IS_BETTER)

    # --- _get_primary_metric ---
    br_acc = BenchmarkResult(
        phase=1, dataset="x", task_type="classify",
        metrics={"accuracy": 0.9, "f1_macro": 0.85},
        duration_seconds=0.1, num_samples=10,
    )
    pm = _get_primary_metric(br_acc)
    check("primary_metric_accuracy", pm is not None and pm[0] == "accuracy")

    br_fs = BenchmarkResult(
        phase=7, dataset="x", task_type="few_shot",
        metrics={"mean_accuracy": 0.8, "ci_95": 0.02},
        duration_seconds=0.1, num_samples=10,
    )
    pm2 = _get_primary_metric(br_fs)
    check("primary_metric_mean_accuracy", pm2 is not None and pm2[0] == "mean_accuracy")

    br_empty = BenchmarkResult(
        phase=1, dataset="x", task_type="classify",
        metrics={}, duration_seconds=0.1, num_samples=10,
    )
    pm3 = _get_primary_metric(br_empty)
    check("primary_metric_empty_none", pm3 is None)

    # --- Reproducibility ---
    cfg_rep = _BenchConfig(batch_size=16, seed=99, n_episodes=10)
    r1a = run_phase(1, cfg_rep, mode="dev")
    r1b = run_phase(1, cfg_rep, mode="dev")
    check("reproducibility_accuracy",
          approx(r1a.metrics["accuracy"], r1b.metrics["accuracy"]))

    # --- All phases complete within timeout ---
    cfg_timeout = _BenchConfig(batch_size=32, seed=42, n_episodes=10)
    t0 = time.time()
    for p in range(1, 8):
        run_phase(p, cfg_timeout, mode="dev")
    elapsed = time.time() - t0
    check("all_phases_under_60s", elapsed < 60.0)

    # Final summary
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"Self-tests: {passed} passed, {failed} failed out of {total}")
    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
    print(f"{'=' * 50}")


# Allow running self-tests via --self-test flag
if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.argv.remove("--self-test")
        _run_self_tests()
    else:
        main()
