"""
Meta-Learning Suite Configuration Template.

Comprehensive PyTorch configuration dataclasses for the meta-learning suite.
Extends brain_ai/config.py with fine-grained control over MAML/MAML++/Reptile
algorithms, episode sampling, checkpointing, and training loops.

Hierarchy:
    MetaLearningFullConfig -> MAMLConfig, MAMLPlusPlusConfig, EpisodeConfig,
                              MetaCheckpointConfig, MetaTrainingConfig

Scale Presets: minimal(), dev(), production_1b(), production_3b(), production_7b()
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import copy
import warnings
import tempfile
import os


# ===========================================================================
#  1. MAMLConfig -- Core MAML Algorithm
# ===========================================================================

@dataclass
class MAMLConfig:
    """Core MAML algorithm configuration.

    Supports maml (second-order), fomaml (first-order), and reptile.
    Backend: "auto" selects torch_func > higher > custom.
    """
    algo: str = "maml"
    inner_steps: int = 5
    inner_lr: float = 0.01
    inner_clip: float = 10.0
    second_order: bool = True
    backend: str = "auto"
    reptile_epsilon: float = 0.1
    reptile_epsilon_decay: float = 0.99
    outer_lr: float = 0.001
    outer_clip: Optional[float] = None

    def __post_init__(self):
        # Auto-set second_order based on algorithm
        if self.algo == "fomaml":
            self.second_order = False
        elif self.algo == "reptile":
            self.second_order = False
        elif self.algo == "maml":
            self.second_order = True

        assert self.algo in ("maml", "fomaml", "reptile"), (
            f"Unknown algo: {self.algo!r}. Must be 'maml', 'fomaml', or 'reptile'."
        )
        assert self.inner_steps >= 1, f"inner_steps must be >= 1, got {self.inner_steps}"
        assert self.inner_lr > 0, f"inner_lr must be positive, got {self.inner_lr}"
        assert self.backend in ("auto", "torch_func", "higher", "custom"), (
            f"Unknown backend: {self.backend!r}"
        )
        assert self.inner_clip > 0, f"inner_clip must be positive, got {self.inner_clip}"
        assert 0 < self.reptile_epsilon <= 1.0, (
            f"reptile_epsilon must be in (0, 1], got {self.reptile_epsilon}"
        )
        assert 0 < self.reptile_epsilon_decay <= 1.0, (
            f"reptile_epsilon_decay must be in (0, 1], got {self.reptile_epsilon_decay}"
        )
        assert self.outer_lr > 0, f"outer_lr must be positive, got {self.outer_lr}"
        if self.outer_clip is not None:
            assert self.outer_clip > 0, f"outer_clip must be positive or None, got {self.outer_clip}"


# ===========================================================================
#  2. MAMLPlusPlusConfig -- MAML++ Enhancements
# ===========================================================================

@dataclass
class MAMLPlusPlusConfig:
    """MAML++ enhancements (Antoniou et al., 2019).

    LSLR: per-layer per-step learned LRs.
    MSL: multi-step loss accumulation (uniform / linear_increase / learned).
    Annealing: derivative-order annealing (first -> second order over epochs).
    BN modes: transductive, per_step, frozen.
    """
    use_lslr: bool = False
    lslr_init_lr: float = 0.01
    lslr_min: float = 1e-6
    lslr_max: float = 1.0
    use_msl: bool = False
    msl_weights: str = "uniform"
    use_annealing: bool = False
    annealing_start_epoch: int = 0
    annealing_end_epoch: Optional[int] = None
    bn_mode: str = "per_step"

    def __post_init__(self):
        assert self.msl_weights in ("uniform", "linear_increase", "learned"), (
            f"Unknown msl_weights: {self.msl_weights!r}"
        )
        assert self.bn_mode in ("transductive", "per_step", "frozen"), (
            f"Unknown bn_mode: {self.bn_mode!r}"
        )
        assert self.lslr_min < self.lslr_max, (
            f"lslr_min ({self.lslr_min}) must be < lslr_max ({self.lslr_max})"
        )
        assert self.lslr_init_lr > 0, f"lslr_init_lr must be positive, got {self.lslr_init_lr}"
        assert self.annealing_start_epoch >= 0, (
            f"annealing_start_epoch must be >= 0, got {self.annealing_start_epoch}"
        )
        if self.annealing_end_epoch is not None:
            assert self.annealing_end_epoch >= self.annealing_start_epoch, (
                f"annealing_end_epoch ({self.annealing_end_epoch}) must be >= "
                f"annealing_start_epoch ({self.annealing_start_epoch})"
            )


# ===========================================================================
#  3. EpisodeConfig -- Episode Sampling
# ===========================================================================

@dataclass
class EpisodeConfig:
    """Episode sampling for N-way K-shot meta-learning.

    Datasets: omniglot (28x28), mini_imagenet (84x84), custom, synthetic.
    mini_imagenet auto-sets resize=84.
    """
    n_way: int = 5
    k_shot: int = 1
    q_query: int = 15
    episodes_per_epoch: int = 600
    eval_episodes: int = 600
    dataset: str = "omniglot"
    data_root: str = "data"
    use_rotations: bool = True
    resize: int = 28
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self):
        assert self.n_way >= 2, f"n_way must be >= 2, got {self.n_way}"
        assert self.k_shot >= 1, f"k_shot must be >= 1, got {self.k_shot}"
        assert self.q_query >= 1, f"q_query must be >= 1, got {self.q_query}"
        assert self.dataset in ("omniglot", "mini_imagenet", "custom", "synthetic"), (
            f"Unknown dataset: {self.dataset!r}"
        )
        assert self.episodes_per_epoch >= 1, (
            f"episodes_per_epoch must be >= 1, got {self.episodes_per_epoch}"
        )
        assert self.eval_episodes >= 1, f"eval_episodes must be >= 1, got {self.eval_episodes}"
        assert self.resize > 0, f"resize must be positive, got {self.resize}"
        if self.dataset == "mini_imagenet":
            self.resize = 84


# ===========================================================================
#  4. MetaCheckpointConfig -- Checkpoint Saving
# ===========================================================================

@dataclass
class MetaCheckpointConfig:
    """Checkpoint saving configuration.

    Naming placeholders: {algo}, {dataset}, {n_way}, {k_shot}, {epoch}.
    """
    save_dir: str = "checkpoints/meta"
    save_every: int = 10
    save_best: bool = True
    save_inner_lrs: bool = True
    save_msl_weights: bool = True
    save_rng_state: bool = True
    save_sampler_config: bool = True
    save_metrics_history: int = 10
    naming_pattern: str = "meta_{algo}_{dataset}_{n_way}w{k_shot}s_epoch{epoch:04d}.pt"

    def __post_init__(self):
        assert self.save_every >= 1, f"save_every must be >= 1, got {self.save_every}"
        assert self.save_metrics_history >= 0, (
            f"save_metrics_history must be >= 0, got {self.save_metrics_history}"
        )

    def format_name(self, algo: str, dataset: str, n_way: int,
                    k_shot: int, epoch: int) -> str:
        """Format checkpoint filename from parameters."""
        return self.naming_pattern.format(
            algo=algo, dataset=dataset, n_way=n_way, k_shot=k_shot, epoch=epoch,
        )

    def get_save_path(self, algo: str, dataset: str, n_way: int,
                      k_shot: int, epoch: int) -> Path:
        """Build full save path for a checkpoint."""
        return Path(self.save_dir) / self.format_name(algo, dataset, n_way, k_shot, epoch)


# ===========================================================================
#  5. MetaTrainingConfig -- Training Loop
# ===========================================================================

@dataclass
class MetaTrainingConfig:
    """Training loop: epochs, eval, early stopping, LR scheduling.

    Schedulers: cosine, step (gamma every step_size epochs), plateau.
    Early stopping: stops after `patience` epochs without min_delta improvement.
    """
    num_epochs: int = 100
    eval_every: int = 5
    log_every: int = 10
    patience: int = 20
    min_delta: float = 0.001
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.5
    warmup_epochs: int = 0
    metrics_path: str = "logs/meta_metrics.json"

    def __post_init__(self):
        assert self.num_epochs >= 1, f"num_epochs must be >= 1, got {self.num_epochs}"
        assert self.eval_every >= 1, f"eval_every must be >= 1, got {self.eval_every}"
        assert self.log_every >= 1, f"log_every must be >= 1, got {self.log_every}"
        assert self.patience >= 1, f"patience must be >= 1, got {self.patience}"
        assert self.min_delta >= 0, f"min_delta must be >= 0, got {self.min_delta}"
        assert self.scheduler_type in ("cosine", "step", "plateau"), (
            f"Unknown scheduler_type: {self.scheduler_type!r}"
        )
        assert self.scheduler_step_size >= 1, (
            f"scheduler_step_size must be >= 1, got {self.scheduler_step_size}"
        )
        assert 0 < self.scheduler_gamma <= 1.0, (
            f"scheduler_gamma must be in (0, 1], got {self.scheduler_gamma}"
        )
        assert self.warmup_epochs >= 0, f"warmup_epochs must be >= 0, got {self.warmup_epochs}"


# ===========================================================================
#  6. MetaLearningFullConfig -- Aggregate
# ===========================================================================

@dataclass
class MetaLearningFullConfig:
    """Full meta-learning config aggregating all sub-configs.

    Provides scale presets: minimal(), dev(), production_1b/3b/7b().
    Cross-config validate() checks logical consistency.
    Serialization: to_dict/from_dict, save/load JSON, clone().
    """
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    maml_plus: MAMLPlusPlusConfig = field(default_factory=MAMLPlusPlusConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    checkpoint: MetaCheckpointConfig = field(default_factory=MetaCheckpointConfig)
    training: MetaTrainingConfig = field(default_factory=MetaTrainingConfig)

    def validate(self):
        """Cross-config validation. Asserts for hard errors, warns for soft issues."""
        if self.maml_plus.use_lslr:
            assert self.maml.inner_steps >= 1, "LSLR requires inner_steps >= 1"
        if self.maml_plus.use_msl:
            assert self.maml.inner_steps >= 1, "MSL requires inner_steps >= 1"
        if self.maml_plus.use_annealing and self.maml.algo != "maml":
            warnings.warn(
                f"Derivative-order annealing is only useful for algo='maml', "
                f"not '{self.maml.algo}'. Annealing will have no effect."
            )
        if self.episode.dataset == "mini_imagenet" and self.episode.resize != 84:
            warnings.warn(
                f"mini-ImageNet standard resize is 84x84, but resize={self.episode.resize}."
            )
        if self.training.warmup_epochs > self.training.num_epochs:
            warnings.warn(
                f"warmup_epochs ({self.training.warmup_epochs}) exceeds "
                f"num_epochs ({self.training.num_epochs})."
            )
        if self.training.eval_every > self.training.num_epochs:
            warnings.warn(
                f"eval_every ({self.training.eval_every}) exceeds "
                f"num_epochs ({self.training.num_epochs})."
            )

    # -- Scale presets -----------------------------------------------------

    @classmethod
    def minimal(cls) -> 'MetaLearningFullConfig':
        """~1M params, for unit tests and CI."""
        return cls(
            maml=MAMLConfig(algo="maml", inner_steps=1, inner_lr=0.01, backend="custom"),
            maml_plus=MAMLPlusPlusConfig(),
            episode=EpisodeConfig(
                n_way=5, k_shot=1, q_query=5,
                episodes_per_epoch=10, eval_episodes=10, dataset="synthetic",
            ),
            checkpoint=MetaCheckpointConfig(save_every=1, save_metrics_history=2),
            training=MetaTrainingConfig(num_epochs=2, eval_every=1, log_every=1, patience=2),
        )

    @classmethod
    def dev(cls) -> 'MetaLearningFullConfig':
        """Dev mode: Conv4, Omniglot, CPU-safe, fast iteration."""
        return cls(
            maml=MAMLConfig(algo="maml", inner_steps=3, inner_lr=0.01),
            maml_plus=MAMLPlusPlusConfig(use_lslr=True),
            episode=EpisodeConfig(
                n_way=5, k_shot=1, q_query=15,
                episodes_per_epoch=100, eval_episodes=100, dataset="omniglot",
            ),
            training=MetaTrainingConfig(num_epochs=50, eval_every=5),
        )

    @classmethod
    def production_1b(cls) -> 'MetaLearningFullConfig':
        """~1B params: MAML, torch_func, mini-ImageNet 5w5s, LSLR+MSL."""
        return cls(
            maml=MAMLConfig(algo="maml", inner_steps=5, inner_lr=0.005, backend="torch_func"),
            maml_plus=MAMLPlusPlusConfig(
                use_lslr=True, use_msl=True,
                msl_weights="linear_increase", bn_mode="per_step",
            ),
            episode=EpisodeConfig(
                n_way=5, k_shot=5, q_query=15,
                episodes_per_epoch=600, eval_episodes=600,
                dataset="mini_imagenet", resize=84,
            ),
            training=MetaTrainingConfig(num_epochs=200, eval_every=10),
        )

    @classmethod
    def production_3b(cls) -> 'MetaLearningFullConfig':
        """~3B params: FOMAML, learned MSL, annealing, per-step BN."""
        return cls(
            maml=MAMLConfig(algo="fomaml", inner_steps=5, inner_lr=0.003, backend="torch_func"),
            maml_plus=MAMLPlusPlusConfig(
                use_lslr=True, use_msl=True, msl_weights="learned",
                use_annealing=True, annealing_start_epoch=50, bn_mode="per_step",
            ),
            episode=EpisodeConfig(
                n_way=5, k_shot=5, q_query=15,
                episodes_per_epoch=600, eval_episodes=600,
                dataset="mini_imagenet", resize=84,
            ),
            training=MetaTrainingConfig(num_epochs=300, eval_every=10),
        )

    @classmethod
    def production_7b(cls) -> 'MetaLearningFullConfig':
        """~7B params: FOMAML, outer clip, frozen BN, long training."""
        return cls(
            maml=MAMLConfig(
                algo="fomaml", inner_steps=5, inner_lr=0.001,
                backend="torch_func", outer_clip=1.0,
            ),
            maml_plus=MAMLPlusPlusConfig(
                use_lslr=True, use_msl=True, msl_weights="learned",
                use_annealing=True, annealing_start_epoch=30,
                annealing_end_epoch=100, bn_mode="frozen",
            ),
            episode=EpisodeConfig(
                n_way=5, k_shot=5, q_query=15,
                episodes_per_epoch=1000, eval_episodes=600,
                dataset="mini_imagenet", resize=84,
            ),
            training=MetaTrainingConfig(num_epochs=500, eval_every=20, patience=50),
        )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        """Convert full config to nested dict via dataclasses.asdict."""
        return {
            'maml': asdict(self.maml),
            'maml_plus': asdict(self.maml_plus),
            'episode': asdict(self.episode),
            'checkpoint': asdict(self.checkpoint),
            'training': asdict(self.training),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MetaLearningFullConfig':
        """Reconstruct from nested dict. Missing keys use defaults."""
        return cls(
            maml=MAMLConfig(**d.get('maml', {})),
            maml_plus=MAMLPlusPlusConfig(**d.get('maml_plus', {})),
            episode=EpisodeConfig(**d.get('episode', {})),
            checkpoint=MetaCheckpointConfig(**d.get('checkpoint', {})),
            training=MetaTrainingConfig(**d.get('training', {})),
        )

    def save(self, path: Union[str, Path]):
        """Save config to JSON file. Creates parent dirs if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MetaLearningFullConfig':
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def clone(self) -> 'MetaLearningFullConfig':
        """Deep copy of this config."""
        return copy.deepcopy(self)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "MetaLearningFullConfig Summary",
            "=" * 40,
            f"  Algorithm:      {self.maml.algo}",
            f"  Inner steps:    {self.maml.inner_steps}",
            f"  Inner LR:       {self.maml.inner_lr}",
            f"  Outer LR:       {self.maml.outer_lr}",
            f"  Second order:   {self.maml.second_order}",
            f"  Backend:        {self.maml.backend}",
            "",
            f"  LSLR:           {self.maml_plus.use_lslr}",
            f"  MSL:            {self.maml_plus.use_msl}",
            f"  Annealing:      {self.maml_plus.use_annealing}",
            f"  BN mode:        {self.maml_plus.bn_mode}",
            "",
            f"  N-way:          {self.episode.n_way}",
            f"  K-shot:         {self.episode.k_shot}",
            f"  Q-query:        {self.episode.q_query}",
            f"  Dataset:        {self.episode.dataset}",
            f"  Episodes/epoch: {self.episode.episodes_per_epoch}",
            "",
            f"  Num epochs:     {self.training.num_epochs}",
            f"  Eval every:     {self.training.eval_every}",
            f"  Patience:       {self.training.patience}",
            f"  Scheduler:      {self.training.scheduler_type}",
            "",
            f"  Save dir:       {self.checkpoint.save_dir}",
            f"  Save every:     {self.checkpoint.save_every}",
            f"  Save best:      {self.checkpoint.save_best}",
        ]
        return "\n".join(lines)


# ===========================================================================
#  Self-Test Suite
# ===========================================================================

def _run_tests():
    """Run all self-test groups. Returns True if all pass."""
    results: List[tuple] = []
    test_num = 0

    def report(name: str, passed: bool, detail: str = ""):
        nonlocal test_num
        test_num += 1
        status = "PASS" if passed else "FAIL"
        suffix = f" -- {detail}" if detail else ""
        results.append((test_num, name, passed, suffix))
        print(f"  [{status}] {test_num:2d}. {name}{suffix}")

    print()
    print("=" * 70)
    print("  Meta-Learning Configuration Template -- Self-Test Suite")
    print("=" * 70)
    print()

    # Test 1: MAMLConfig defaults
    try:
        cfg = MAMLConfig()
        checks = [
            cfg.algo == "maml", cfg.inner_steps == 5, cfg.inner_lr == 0.01,
            cfg.inner_clip == 10.0, cfg.second_order is True, cfg.backend == "auto",
            cfg.reptile_epsilon == 0.1, cfg.reptile_epsilon_decay == 0.99,
            cfg.outer_lr == 0.001, cfg.outer_clip is None,
        ]
        report("MAMLConfig defaults are correct",
               all(checks), f"{sum(checks)}/{len(checks)} fields correct")
    except Exception as e:
        report("MAMLConfig defaults are correct", False, str(e))

    # Test 2: MAMLConfig auto-sets second_order based on algo
    try:
        m = MAMLConfig(algo="maml")
        f = MAMLConfig(algo="fomaml")
        r = MAMLConfig(algo="reptile")
        checks = [m.second_order is True, f.second_order is False, r.second_order is False]
        report("MAMLConfig auto-sets second_order based on algo",
               all(checks), f"maml={m.second_order}, fomaml={f.second_order}, reptile={r.second_order}")
    except Exception as e:
        report("MAMLConfig auto-sets second_order based on algo", False, str(e))

    # Test 3: MAMLConfig validation rejects invalid algo
    try:
        rejected = False
        try:
            MAMLConfig(algo="invalid_algo")
        except (AssertionError, ValueError):
            rejected = True
        report("MAMLConfig validation rejects invalid algo",
               rejected, "AssertionError raised" if rejected else "No error raised")
    except Exception as e:
        report("MAMLConfig validation rejects invalid algo", False, str(e))

    # Test 4: MAMLPlusPlusConfig defaults
    try:
        cfg = MAMLPlusPlusConfig()
        checks = [
            cfg.use_lslr is False, cfg.lslr_init_lr == 0.01,
            cfg.lslr_min == 1e-6, cfg.lslr_max == 1.0,
            cfg.use_msl is False, cfg.msl_weights == "uniform",
            cfg.use_annealing is False, cfg.annealing_start_epoch == 0,
            cfg.annealing_end_epoch is None, cfg.bn_mode == "per_step",
        ]
        report("MAMLPlusPlusConfig defaults are correct",
               all(checks), f"{sum(checks)}/{len(checks)} fields correct")
    except Exception as e:
        report("MAMLPlusPlusConfig defaults are correct", False, str(e))

    # Test 5: MAMLPlusPlusConfig validation rejects invalid values
    try:
        r_msl = r_bn = r_lr = False
        try:
            MAMLPlusPlusConfig(msl_weights="invalid_weights")
        except (AssertionError, ValueError):
            r_msl = True
        try:
            MAMLPlusPlusConfig(bn_mode="invalid_bn")
        except (AssertionError, ValueError):
            r_bn = True
        try:
            MAMLPlusPlusConfig(lslr_min=1.0, lslr_max=0.001)
        except (AssertionError, ValueError):
            r_lr = True
        report("MAMLPlusPlusConfig validation rejects invalid values",
               r_msl and r_bn and r_lr, f"msl={r_msl}, bn={r_bn}, lr_range={r_lr}")
    except Exception as e:
        report("MAMLPlusPlusConfig validation rejects invalid values", False, str(e))

    # Test 6: EpisodeConfig defaults
    try:
        cfg = EpisodeConfig()
        checks = [
            cfg.n_way == 5, cfg.k_shot == 1, cfg.q_query == 15,
            cfg.episodes_per_epoch == 600, cfg.eval_episodes == 600,
            cfg.dataset == "omniglot", cfg.data_root == "data",
            cfg.use_rotations is True, cfg.resize == 28,
            cfg.seed == 42, cfg.num_workers == 0,
        ]
        report("EpisodeConfig defaults are correct",
               all(checks), f"{sum(checks)}/{len(checks)} fields correct")
    except Exception as e:
        report("EpisodeConfig defaults are correct", False, str(e))

    # Test 7: EpisodeConfig auto-sets resize for mini_imagenet
    try:
        c_mi = EpisodeConfig(dataset="mini_imagenet")
        c_om = EpisodeConfig(dataset="omniglot")
        c_sy = EpisodeConfig(dataset="synthetic", resize=32)
        checks = [c_mi.resize == 84, c_mi.dataset == "mini_imagenet",
                  c_om.resize == 28, c_sy.resize == 32]
        report("EpisodeConfig auto-sets resize for mini_imagenet",
               all(checks),
               f"mini_imagenet={c_mi.resize}, omniglot={c_om.resize}, synthetic={c_sy.resize}")
    except Exception as e:
        report("EpisodeConfig auto-sets resize for mini_imagenet", False, str(e))

    # Test 8: MetaCheckpointConfig defaults
    try:
        cfg = MetaCheckpointConfig()
        checks = [
            cfg.save_dir == "checkpoints/meta", cfg.save_every == 10,
            cfg.save_best is True, cfg.save_inner_lrs is True,
            cfg.save_msl_weights is True, cfg.save_rng_state is True,
            cfg.save_sampler_config is True, cfg.save_metrics_history == 10,
            "{algo}" in cfg.naming_pattern, "{epoch" in cfg.naming_pattern,
        ]
        report("MetaCheckpointConfig defaults are correct",
               all(checks), f"{sum(checks)}/{len(checks)} fields correct")
    except Exception as e:
        report("MetaCheckpointConfig defaults are correct", False, str(e))

    # Test 9: MetaTrainingConfig defaults
    try:
        cfg = MetaTrainingConfig()
        checks = [
            cfg.num_epochs == 100, cfg.eval_every == 5, cfg.log_every == 10,
            cfg.patience == 20, cfg.min_delta == 0.001, cfg.use_scheduler is True,
            cfg.scheduler_type == "cosine", cfg.scheduler_step_size == 30,
            cfg.scheduler_gamma == 0.5, cfg.warmup_epochs == 0,
            cfg.metrics_path == "logs/meta_metrics.json",
        ]
        report("MetaTrainingConfig defaults are correct",
               all(checks), f"{sum(checks)}/{len(checks)} fields correct")
    except Exception as e:
        report("MetaTrainingConfig defaults are correct", False, str(e))

    # Test 10: MetaLearningFullConfig.minimal()
    try:
        cfg = MetaLearningFullConfig.minimal()
        checks = [
            cfg.maml.algo == "maml", cfg.maml.inner_steps == 1,
            cfg.maml.backend == "custom", cfg.episode.dataset == "synthetic",
            cfg.episode.episodes_per_epoch == 10, cfg.training.num_epochs == 2,
            cfg.maml_plus.use_lslr is False, cfg.maml_plus.use_msl is False,
        ]
        report("MetaLearningFullConfig.minimal() creates valid config",
               all(checks), f"{sum(checks)}/{len(checks)} checks passed")
    except Exception as e:
        report("MetaLearningFullConfig.minimal() creates valid config", False, str(e))

    # Test 11: MetaLearningFullConfig.dev()
    try:
        cfg = MetaLearningFullConfig.dev()
        checks = [
            cfg.maml.algo == "maml", cfg.maml.inner_steps == 3,
            cfg.maml_plus.use_lslr is True, cfg.episode.dataset == "omniglot",
            cfg.episode.n_way == 5, cfg.episode.k_shot == 1,
            cfg.training.num_epochs == 50,
        ]
        report("MetaLearningFullConfig.dev() creates valid config",
               all(checks), f"{sum(checks)}/{len(checks)} checks passed")
    except Exception as e:
        report("MetaLearningFullConfig.dev() creates valid config", False, str(e))

    # Test 12: Production presets (1b/3b/7b)
    try:
        c1 = MetaLearningFullConfig.production_1b()
        c3 = MetaLearningFullConfig.production_3b()
        c7 = MetaLearningFullConfig.production_7b()
        ch1 = [
            c1.maml.algo == "maml", c1.maml.inner_steps == 5,
            c1.maml.inner_lr == 0.005, c1.maml.backend == "torch_func",
            c1.maml_plus.use_lslr is True, c1.maml_plus.use_msl is True,
            c1.maml_plus.msl_weights == "linear_increase",
            c1.episode.dataset == "mini_imagenet", c1.episode.resize == 84,
        ]
        ch3 = [
            c3.maml.algo == "fomaml", c3.maml.second_order is False,
            c3.maml_plus.msl_weights == "learned",
            c3.maml_plus.use_annealing is True, c3.training.num_epochs == 300,
        ]
        ch7 = [
            c7.maml.algo == "fomaml", c7.maml.outer_clip == 1.0,
            c7.maml_plus.bn_mode == "frozen", c7.maml_plus.annealing_end_epoch == 100,
            c7.episode.episodes_per_epoch == 1000,
            c7.training.num_epochs == 500, c7.training.patience == 50,
        ]
        ok = all(ch1) and all(ch3) and all(ch7)
        total = len(ch1) + len(ch3) + len(ch7)
        p = sum(ch1) + sum(ch3) + sum(ch7)
        report("Production presets (1b/3b/7b) all valid",
               ok, f"{p}/{total} checks (1b={sum(ch1)}/{len(ch1)}, "
                    f"3b={sum(ch3)}/{len(ch3)}, 7b={sum(ch7)}/{len(ch7)})")
    except Exception as e:
        report("Production presets (1b/3b/7b) all valid", False, str(e))

    # Test 13: Cross-config validation detects issues
    try:
        # Annealing warning for FOMAML
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = MetaLearningFullConfig(
                maml=MAMLConfig(algo="fomaml"),
                maml_plus=MAMLPlusPlusConfig(use_annealing=True),
            )
            cfg.validate()
            annealing_warned = any("annealing" in str(x.message).lower() for x in w)

        # mini-ImageNet resize warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg2 = MetaLearningFullConfig(episode=EpisodeConfig(dataset="omniglot"))
            cfg2.episode.dataset = "mini_imagenet"
            cfg2.episode.resize = 28
            cfg2.validate()
            resize_warned = any("resize" in str(x.message).lower() or "84" in str(x.message) for x in w)

        # Warmup exceeds epochs
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg3 = MetaLearningFullConfig(
                training=MetaTrainingConfig(num_epochs=10, warmup_epochs=20))
            cfg3.validate()
            warmup_warned = any("warmup" in str(x.message).lower() for x in w)

        # eval_every exceeds epochs
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg4 = MetaLearningFullConfig(
                training=MetaTrainingConfig(num_epochs=5, eval_every=10))
            cfg4.validate()
            eval_warned = any("eval_every" in str(x.message).lower() for x in w)

        all_det = annealing_warned and resize_warned and warmup_warned and eval_warned
        report("Cross-config validation detects issues", all_det,
               f"annealing={annealing_warned}, resize={resize_warned}, "
               f"warmup={warmup_warned}, eval={eval_warned}")
    except Exception as e:
        report("Cross-config validation detects issues", False, str(e))

    # Test 14: Serialization round-trip (to_dict -> from_dict -> to_dict)
    try:
        presets = [
            ("minimal", MetaLearningFullConfig.minimal()),
            ("dev", MetaLearningFullConfig.dev()),
            ("prod_1b", MetaLearningFullConfig.production_1b()),
            ("prod_3b", MetaLearningFullConfig.production_3b()),
            ("prod_7b", MetaLearningFullConfig.production_7b()),
        ]
        all_ok = True
        failed = []
        for name, cfg in presets:
            d1 = cfg.to_dict()
            d2 = MetaLearningFullConfig.from_dict(d1).to_dict()
            if d1 != d2:
                all_ok = False
                failed.append(name)
        report("Serialization round-trip: to_dict -> from_dict -> to_dict",
               all_ok,
               f"All {len(presets)} presets round-trip correctly" if all_ok
               else f"Failed: {', '.join(failed)}")
    except Exception as e:
        report("Serialization round-trip: to_dict -> from_dict -> to_dict", False, str(e))

    # Test 15: JSON save/load round-trip
    try:
        all_ok = True
        failed = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, cfg in presets:
                path = os.path.join(tmpdir, f"{name}.json")
                cfg.save(path)
                assert os.path.exists(path), f"{path} not created"
                with open(path) as fh:
                    raw = json.load(fh)
                assert isinstance(raw, dict), "JSON root must be dict"
                loaded = MetaLearningFullConfig.load(path)
                if cfg.to_dict() != loaded.to_dict():
                    all_ok = False
                    failed.append(name)
        report("JSON save/load round-trip", all_ok,
               f"All {len(presets)} presets JSON round-trip correctly" if all_ok
               else f"Failed: {', '.join(failed)}")
    except Exception as e:
        report("JSON save/load round-trip", False, str(e))

    # Test 16: All presets pass validate()
    try:
        preset_methods = [
            ("minimal", MetaLearningFullConfig.minimal),
            ("dev", MetaLearningFullConfig.dev),
            ("production_1b", MetaLearningFullConfig.production_1b),
            ("production_3b", MetaLearningFullConfig.production_3b),
            ("production_7b", MetaLearningFullConfig.production_7b),
        ]
        all_valid = True
        errs = []
        for name, factory in preset_methods:
            try:
                cfg = factory()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cfg.validate()
            except Exception as exc:
                all_valid = False
                errs.append(f"{name}: {exc}")
        report("All presets pass validate()", all_valid,
               f"All {len(preset_methods)} presets pass" if all_valid
               else f"Failures: {'; '.join(errs)}")
    except Exception as e:
        report("All presets pass validate()", False, str(e))

    # Test 17: Clone produces independent copy
    try:
        original = MetaLearningFullConfig.dev()
        cloned = original.clone()
        assert original.to_dict() == cloned.to_dict(), "Clone should equal original"
        cloned.maml.inner_lr = 0.999
        cloned.episode.n_way = 20
        cloned.training.num_epochs = 9999
        cloned.maml_plus.use_lslr = not original.maml_plus.use_lslr
        cloned.checkpoint.save_every = 999
        checks = [
            original.maml.inner_lr != cloned.maml.inner_lr,
            original.episode.n_way != cloned.episode.n_way,
            original.training.num_epochs != cloned.training.num_epochs,
            original.maml_plus.use_lslr != cloned.maml_plus.use_lslr,
            original.checkpoint.save_every != cloned.checkpoint.save_every,
        ]
        report("Clone produces independent copy",
               all(checks), f"{sum(checks)}/{len(checks)} independence checks passed")
    except Exception as e:
        report("Clone produces independent copy", False, str(e))

    # Test 18: Summary generates non-empty string
    try:
        for name, factory in preset_methods:
            cfg = factory()
            s = cfg.summary()
            assert isinstance(s, str) and len(s) > 100, f"summary too short for {name}"
            assert "Algorithm" in s, "summary must contain 'Algorithm'"
            assert cfg.maml.algo in s, f"summary must contain '{cfg.maml.algo}'"
        report("Summary generates non-empty string", True,
               f"All {len(preset_methods)} presets produce valid summaries")
    except Exception as e:
        report("Summary generates non-empty string", False, str(e))

    # Test 19: Checkpoint naming format
    try:
        ckpt = MetaCheckpointConfig()
        name = ckpt.format_name(algo="maml", dataset="omniglot", n_way=5, k_shot=1, epoch=42)
        checks = [
            "maml" in name, "omniglot" in name, "5w" in name,
            "1s" in name, "0042" in name, name.endswith(".pt"),
        ]
        path = ckpt.get_save_path(algo="fomaml", dataset="mini_imagenet",
                                  n_way=5, k_shot=5, epoch=100)
        checks += [
            isinstance(path, Path), "checkpoints/meta" in str(path),
            "fomaml" in str(path), "0100" in str(path),
        ]
        report("Checkpoint naming format works correctly",
               all(checks), f"{sum(checks)}/{len(checks)} format checks passed")
    except Exception as e:
        report("Checkpoint naming format works correctly", False, str(e))

    # Test 20: EpisodeConfig validation rejects bad values
    try:
        r_nw = r_ks = r_qq = r_ds = False
        try:
            EpisodeConfig(n_way=1)
        except (AssertionError, ValueError):
            r_nw = True
        try:
            EpisodeConfig(k_shot=0)
        except (AssertionError, ValueError):
            r_ks = True
        try:
            EpisodeConfig(q_query=0)
        except (AssertionError, ValueError):
            r_qq = True
        try:
            EpisodeConfig(dataset="nonexistent")
        except (AssertionError, ValueError):
            r_ds = True
        report("EpisodeConfig validation rejects bad values",
               r_nw and r_ks and r_qq and r_ds,
               f"n_way={r_nw}, k_shot={r_ks}, q_query={r_qq}, dataset={r_ds}")
    except Exception as e:
        report("EpisodeConfig validation rejects bad values", False, str(e))

    # -- Summary -----------------------------------------------------------
    print()
    print("-" * 70)
    total = len(results)
    passed = sum(1 for _, _, p, _ in results if p)
    failed = total - passed
    print(f"  Results: {passed}/{total} PASSED, {failed} FAILED")
    if failed > 0:
        print("\n  Failed tests:")
        for num, name, p, detail in results:
            if not p:
                print(f"    {num:2d}. {name}{detail}")
    print("-" * 70)
    print()
    return failed == 0


# ===========================================================================
#  Entry Point
# ===========================================================================

if __name__ == "__main__":
    success = _run_tests()
    if not success:
        raise SystemExit(1)
