"""
scaffold_generator.py
=====================
Main scaffold generator for the world model project structure.

Creates a complete directory tree with base classes, world model composition
root, YAML configs, pyproject.toml, and README.md.

Usage (self-test):
    python scaffold_generator.py

Usage (generation):
    from scaffold_generator import ScaffoldGenerator
    gen = ScaffoldGenerator()
    gen.generate("/path/to/output", project_name="my_world_model")
"""

from __future__ import annotations

import ast
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Directory / file constants
# ---------------------------------------------------------------------------

SUBPACKAGES = [
    "encoders",
    "dynamics",
    "memory",
    "planning",
    "decoders",
    "training",
    "evaluation",
    "deployment",
    "core",
]

CONFIG_FILES = ["model.yaml", "training.yaml", "dataset.yaml", "hardware.yaml"]


# ---------------------------------------------------------------------------
# __init__.py content per subpackage
# ---------------------------------------------------------------------------

INIT_TEMPLATES: Dict[str, str] = {
    "root": """\
\"\"\"World model package root.\"\"\"
from .core.world_model import BaseWorldModel

__all__ = ["BaseWorldModel"]
""",
    "encoders": """\
\"\"\"Observation encoder base classes and registry.\"\"\"
from .base import BaseEncoder

__all__ = ["BaseEncoder"]
""",
    "dynamics": """\
\"\"\"Dynamics model base classes and registry.\"\"\"
from .base import BaseDynamics

__all__ = ["BaseDynamics"]
""",
    "memory": """\
\"\"\"Memory component base classes.\"\"\"
from .base import BaseMemory

__all__ = ["BaseMemory"]
""",
    "planning": """\
\"\"\"Planner base classes.\"\"\"
from .base import BasePlanner

__all__ = ["BasePlanner"]
""",
    "decoders": """\
\"\"\"Observation decoder base classes.\"\"\"
from .base import BaseDecoder

__all__ = ["BaseDecoder"]
""",
    "training": """\
\"\"\"Training orchestrator base classes.\"\"\"
from .base import BaseTrainer

__all__ = ["BaseTrainer"]
""",
    "evaluation": """\
\"\"\"Evaluation harness base classes.\"\"\"
from .base import BaseEvaluator

__all__ = ["BaseEvaluator"]
""",
    "deployment": """\
\"\"\"Deployment exporter base classes.\"\"\"
from .base import BaseExporter

__all__ = ["BaseExporter"]
""",
    "core": """\
\"\"\"Composition root: BaseWorldModel.\"\"\"
from .world_model import BaseWorldModel

__all__ = ["BaseWorldModel"]
""",
}


# ---------------------------------------------------------------------------
# base.py content per subpackage (condensed from base_classes_template)
# ---------------------------------------------------------------------------

BASE_TEMPLATES: Dict[str, str] = {
    "encoders": '''\
"""BaseEncoder — abstract base class for observation encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class BaseEncoder(ABC):
    """Map raw observations to a fixed-dimensional latent embedding.

    Subclasses must also inherit nn.Module for parameter registration::

        class MyEncoder(nn.Module, BaseEncoder): ...

    All implementations must be stateless: the same input always produces
    the same output given the same parameters.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of observations.

        Parameters
        ----------
        x : Tensor
            Shape (B, *obs_shape).

        Returns
        -------
        Tensor
            Shape (B, embed_dim).
        """

    @abstractmethod
    def get_embed_dim(self) -> int:
        """Return the embedding dimensionality (strictly positive int)."""

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return the full output shape excluding batch dimension."""
''',
    "dynamics": '''\
"""BaseDynamics — abstract base class for latent-space dynamics models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


class BaseDynamics(ABC):
    """Predict next latent state from (state, action) pairs.

    Dimensional contract: get_state_dim() must equal the paired
    encoder's get_embed_dim(). BaseWorldModel validates this.
    """

    @abstractmethod
    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict next latent state.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).
        action : Tensor
            Shape (B, action_dim).

        Returns
        -------
        Tensor
            Shape (B, state_dim).
        """

    @abstractmethod
    def imagine(
        self, state: Tensor, policy: Callable[[Tensor], Tensor], horizon: int
    ) -> Tensor:
        """Roll out an imagined trajectory.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).
        policy : Callable[[Tensor], Tensor]
            Maps (B, state_dim) -> (B, action_dim).
        horizon : int
            Steps to imagine (>= 1).

        Returns
        -------
        Tensor
            Shape (B, horizon, state_dim).
        """

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return the latent state dimensionality (strictly positive int)."""
''',
    "memory": '''\
"""BaseMemory — abstract base class for episodic/working memory."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseMemory(ABC):
    """Stateful memory that accumulates key-value pairs.

    Must be reset between episodes. Concurrent access requires
    external locking.
    """

    @abstractmethod
    def read(self, query: Tensor) -> Tensor:
        """Retrieve memory contents. Returns zeros when empty.

        Parameters
        ----------
        query : Tensor
            Shape (B, query_dim).

        Returns
        -------
        Tensor
            Shape (B, value_dim).
        """

    @abstractmethod
    def write(self, key: Tensor, value: Tensor) -> None:
        """Store a key-value pair.

        Parameters
        ----------
        key : Tensor
            Shape (B, key_dim).
        value : Tensor
            Shape (B, value_dim).
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memory (does not reset learnable params)."""

    @abstractmethod
    def get_capacity(self) -> int:
        """Return max capacity, or -1 for unlimited."""
''',
    "planning": '''\
"""BasePlanner — abstract base class for model-based planners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from world_model.dynamics.base import BaseDynamics


class BasePlanner(ABC):
    """Compute action sequences via latent-space simulation.

    Planners do not interact with the real environment — they use
    a dynamics model to simulate future states.
    """

    @abstractmethod
    def plan(
        self, state: Tensor, dynamics: BaseDynamics, horizon: int
    ) -> Tensor:
        """Compute an optimized action sequence.

        Parameters
        ----------
        state : Tensor
            Shape (B, state_dim).
        dynamics : BaseDynamics
            Dynamics model to simulate with.
        horizon : int
            Steps to plan (>= 1).

        Returns
        -------
        Tensor
            Shape (B, horizon, action_dim).
        """
''',
    "decoders": '''\
"""BaseDecoder — abstract base class for latent-to-observation decoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class BaseDecoder(ABC):
    """Map latent vectors back to observation space.

    Dimensional contract: get_input_dim() must equal the paired
    encoder's get_embed_dim(). BaseWorldModel validates this.
    """

    @abstractmethod
    def forward(self, latent: Tensor) -> Tensor:
        """Decode a batch of latent vectors.

        Parameters
        ----------
        latent : Tensor
            Shape (B, input_dim).

        Returns
        -------
        Tensor
            Shape (B, *output_shape).
        """

    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """Return decoded observation shape (no batch dim)."""

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return expected latent dimensionality (strictly positive int)."""
''',
    "training": '''\
"""BaseTrainer — abstract base class for training orchestrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Encapsulate one gradient update step, validation, and checkpointing.

    Use module.train(False) to set inference mode in validate(); do not
    call the bare inference toggle method directly.
    """

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """Run one forward-backward-update cycle. Must include 'loss' key."""

    @abstractmethod
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Compute validation metrics without parameter updates."""

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Persist model and optimizer state to path."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from path."""
''',
    "evaluation": '''\
"""BaseEvaluator — abstract base class for evaluation harnesses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader


class BaseEvaluator(ABC):
    """Compute evaluation metrics on a trained model.

    Use module.train(False) inside evaluate(); do not call the bare
    inference toggle method directly.
    """

    @abstractmethod
    def evaluate(self, model: Any, dataloader: DataLoader) -> Dict[str, float]:
        """Run full evaluation over the dataloader."""

    @abstractmethod
    def compute_metrics(
        self, predictions: Tensor, targets: Tensor
    ) -> Dict[str, float]:
        """Compute metric values. Raises ValueError on shape mismatch."""
''',
    "deployment": '''\
"""BaseExporter — abstract base class for model exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExporter(ABC):
    """Export trained models to deployment formats and validate artifacts."""

    @abstractmethod
    def export(self, model: Any, path: str, format: str) -> None:
        """Serialize model to path in the given format."""

    @abstractmethod
    def validate_export(self, path: str) -> bool:
        """Check artifact validity. Returns False (never raises) on failure."""
''',
}


# ---------------------------------------------------------------------------
# ScaffoldGenerator
# ---------------------------------------------------------------------------


class ScaffoldGenerator:
    """Generate the complete world model project structure.

    Parameters
    ----------
    verbose : bool
        Print each created file to stdout.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._created: List[str] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate(
        self,
        output_dir: str,
        project_name: str = "world_model",
        skip_configs: bool = False,
        skip_readme: bool = False,
    ) -> None:
        """Create the full project scaffold at output_dir.

        Parameters
        ----------
        output_dir : str
            Root directory for the generated project.
        project_name : str
            PyPI / module name (used in pyproject.toml).
        skip_configs : bool
            If True, do not write YAML config files.
        skip_readme : bool
            If True, do not write README.md.
        """
        self._created = []
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        pkg = root / "world_model"

        self._create_package_tree(pkg)
        self._write_pyproject_toml(root, project_name)
        if not skip_configs:
            self._write_configs(pkg / "configs")
        if not skip_readme:
            self._write_readme(root, project_name)

        if self.verbose:
            print(f"\nScaffold generated: {len(self._created)} files written to {output_dir}")

    def validate_scaffold(self, output_dir: str) -> bool:
        """Verify the scaffold at output_dir is complete.

        Returns True if all required files exist and are non-empty.
        """
        root = Path(output_dir)
        pkg = root / "world_model"

        required: List[Path] = []

        # Root package
        required.append(pkg / "__init__.py")

        # Subpackages
        for sub in SUBPACKAGES:
            required.append(pkg / sub / "__init__.py")
            if sub != "core":
                required.append(pkg / sub / "base.py")

        # Core world model
        required.append(pkg / "core" / "world_model.py")

        # Configs
        for cfg in CONFIG_FILES:
            required.append(pkg / "configs" / cfg)

        # Root files
        required.append(root / "pyproject.toml")

        all_ok = True
        for path in required:
            exists = path.is_file() and path.stat().st_size > 0
            if not exists:
                if self.verbose:
                    print(f"  MISSING: {path}")
                all_ok = False

        return all_ok

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _write(self, path: Path, content: str) -> None:
        """Write content to path, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._created.append(str(path))
        if self.verbose:
            print(f"  wrote: {path}")

    def _create_package_tree(self, pkg: Path) -> None:
        """Create all subpackage __init__.py and base.py files."""
        # Root package __init__.py
        self._write(pkg / "__init__.py", INIT_TEMPLATES["root"])

        for sub in SUBPACKAGES:
            init_content = INIT_TEMPLATES.get(sub, f'"""{sub} subpackage."""\n')
            self._write(pkg / sub / "__init__.py", init_content)

        # base.py for each non-core subpackage
        for sub in SUBPACKAGES:
            if sub == "core":
                self._write(
                    pkg / "core" / "world_model.py",
                    self._get_world_model_content(),
                )
            else:
                base_content = BASE_TEMPLATES.get(sub, f'"""{sub} base module."""\n')
                self._write(pkg / sub / "base.py", base_content)

    def _write_configs(self, configs_dir: Path) -> None:
        """Write the four YAML config files."""
        # Import here to allow standalone usage without installed package
        _this_dir = Path(__file__).parent
        sys.path.insert(0, str(_this_dir))
        try:
            from config_templates import (
                generate_model_config,
                generate_training_config,
                generate_dataset_config,
                generate_hardware_config,
            )
        except ImportError:
            # Fallback: write minimal placeholder configs
            self._write_minimal_configs(configs_dir)
            return

        configs = {
            "model.yaml": generate_model_config(),
            "training.yaml": generate_training_config(),
            "dataset.yaml": generate_dataset_config(),
            "hardware.yaml": generate_hardware_config(),
        }
        for fname, content in configs.items():
            self._write(configs_dir / fname, content)

    def _write_minimal_configs(self, configs_dir: Path) -> None:
        """Write minimal placeholder configs when config_templates is unavailable."""
        placeholders = {
            "model.yaml": "# model config\nencoder:\n  type: vit\ndynamics:\n  type: rssm\ndecoder:\n  type: conv\n",
            "training.yaml": "# training config\noptimizer:\n  type: adamw\n  lr: 3.0e-4\ntraining:\n  epochs: 100\n  batch_size: 32\n",
            "dataset.yaml": "# dataset config\ndata:\n  root: /data\nsplits:\n  train: train/\n  val: val/\nsequence:\n  length: 16\n",
            "hardware.yaml": "# hardware config\ndevice: cuda\nprecision:\n  dtype: float32\n",
        }
        for fname, content in placeholders.items():
            self._write(configs_dir / fname, content)

    def _write_pyproject_toml(self, root: Path, project_name: str) -> None:
        """Write pyproject.toml to root."""
        _this_dir = Path(__file__).parent
        sys.path.insert(0, str(_this_dir))
        try:
            from pyproject_template import generate_pyproject_toml
            content = generate_pyproject_toml(project_name=project_name)
        except ImportError:
            content = self._minimal_pyproject(project_name)
        self._write(root / "pyproject.toml", content)

    def _minimal_pyproject(self, project_name: str) -> str:
        return f"""\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "{project_name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch>=2.2", "einops>=0.7", "pyyaml>=6.0", "numpy>=1.24"]

[tool.setuptools.packages.find]
where = ["."]
include = ["world_model*"]
"""

    def _write_readme(self, root: Path, project_name: str) -> None:
        """Write README.md to root."""
        content = f"""\
# {project_name}

Modular world model framework with hot-swappable components.

## Architecture

```
world_model/
├── encoders/      # BaseEncoder — observation to latent
├── dynamics/      # BaseDynamics — latent state transitions
├── memory/        # BaseMemory — episodic memory
├── planning/      # BasePlanner — model-based action planning
├── decoders/      # BaseDecoder — latent to observation
├── training/      # BaseTrainer — training loop and checkpointing
├── evaluation/    # BaseEvaluator — evaluation harness
├── deployment/    # BaseExporter — model export
├── core/          # BaseWorldModel — composition root
└── configs/       # YAML configuration files
```

## Design Philosophy

Every major component implements an ABC interface contract.
`BaseWorldModel` composes them via dependency injection — pass
different encoder/dynamics/decoder instances to swap implementations
without touching any other code.

```python
from world_model.core.world_model import BaseWorldModel

model = BaseWorldModel(
    encoder=ViTEncoder(embed_dim=512),
    dynamics=RSSMDynamics(state_dim=512, action_dim=6),
    decoder=ConvDecoder(input_dim=512),
)
```

## Hot-Swap

```python
# Replace ViT with Mamba — zero changes to dynamics or decoder
model = BaseWorldModel(
    encoder=MambaEncoder(embed_dim=512),
    dynamics=RSSMDynamics(state_dim=512, action_dim=6),
    decoder=ConvDecoder(input_dim=512),
)
```

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest tests/ -v
```
"""
        self._write(root / "README.md", content)

    def _get_world_model_content(self) -> str:
        """Return the BaseWorldModel source for core/world_model.py."""
        return '''\
"""BaseWorldModel — composition root for the world model scaffold."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from world_model.encoders.base import BaseEncoder
from world_model.dynamics.base import BaseDynamics
from world_model.memory.base import BaseMemory
from world_model.planning.base import BasePlanner
from world_model.decoders.base import BaseDecoder


class BaseWorldModel(nn.Module):
    """Composition root for a world model.

    Accepts encoder, dynamics, decoder, and optional memory and planner
    as constructor arguments. Validates dimensional compatibility at
    construction time. Delegates all computation to components.

    Parameters
    ----------
    encoder : BaseEncoder
        Maps observations to latent vectors.
    dynamics : BaseDynamics
        Predicts next latent state.
    decoder : BaseDecoder
        Reconstructs observations from latent vectors.
    memory : Optional[BaseMemory]
        Episodic memory; if provided, step() reads context before dynamics.
    planner : Optional[BasePlanner]
        Action planner; accessible via plan().
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        dynamics: BaseDynamics,
        decoder: BaseDecoder,
        memory: Optional[BaseMemory] = None,
        planner: Optional[BasePlanner] = None,
    ) -> None:
        super().__init__()

        if not isinstance(encoder, BaseEncoder):
            raise TypeError(
                f"encoder must be BaseEncoder, got {type(encoder).__name__}"
            )
        if not isinstance(dynamics, BaseDynamics):
            raise TypeError(
                f"dynamics must be BaseDynamics, got {type(dynamics).__name__}"
            )
        if not isinstance(decoder, BaseDecoder):
            raise TypeError(
                f"decoder must be BaseDecoder, got {type(decoder).__name__}"
            )
        if memory is not None and not isinstance(memory, BaseMemory):
            raise TypeError(
                f"memory must be BaseMemory or None, got {type(memory).__name__}"
            )
        if planner is not None and not isinstance(planner, BasePlanner):
            raise TypeError(
                f"planner must be BasePlanner or None, got {type(planner).__name__}"
            )

        enc_dim = encoder.get_embed_dim()
        dyn_dim = dynamics.get_state_dim()
        dec_dim = decoder.get_input_dim()

        if enc_dim <= 0:
            raise ValueError(f"encoder.get_embed_dim() must be > 0, got {enc_dim}")
        if enc_dim != dyn_dim:
            raise ValueError(
                f"Dimension mismatch: encoder.get_embed_dim()={enc_dim} != "
                f"dynamics.get_state_dim()={dyn_dim}"
            )
        if enc_dim != dec_dim:
            raise ValueError(
                f"Dimension mismatch: encoder.get_embed_dim()={enc_dim} != "
                f"decoder.get_input_dim()={dec_dim}"
            )

        self.encoder = encoder  # type: ignore[assignment]
        self.dynamics = dynamics  # type: ignore[assignment]
        self.decoder = decoder  # type: ignore[assignment]

        self._has_memory = memory is not None
        self._has_planner = planner is not None
        self._embed_dim = enc_dim

        if memory is not None:
            self.memory = memory  # type: ignore[assignment]
        if planner is not None:
            self.planner = planner  # type: ignore[assignment]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def state_dim(self) -> int:
        return self._embed_dim

    @property
    def has_memory(self) -> bool:
        return self._has_memory

    @property
    def has_planner(self) -> bool:
        return self._has_planner

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self.decoder.get_output_shape()

    def encode(self, obs: Tensor) -> Tensor:
        return self.encoder.forward(obs)

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        if self._has_memory:
            context = self.memory.read(state)
            augmented = state + context
            next_state = self.dynamics.step(augmented, action)
            self.memory.write(state, next_state)
        else:
            next_state = self.dynamics.step(state, action)
        return next_state

    def imagine(
        self, state: Tensor, policy: Callable[[Tensor], Tensor], horizon: int
    ) -> Tensor:
        return self.dynamics.imagine(state, policy, horizon)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder.forward(latent)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        latent = self.encode(obs)
        next_state = self.step(latent, action)
        return self.decode(next_state)

    def reset_memory(self) -> None:
        if self._has_memory:
            self.memory.reset()

    def plan(self, state: Tensor, horizon: int) -> Tensor:
        if not self._has_planner:
            raise RuntimeError(
                "plan() called on BaseWorldModel without a planner. "
                "Pass a BasePlanner instance at construction time."
            )
        return self.planner.plan(state, self.dynamics, horizon)
'''


# ---------------------------------------------------------------------------
# Self-test block
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import tempfile

    errors: list[str] = []

    def fail(msg: str) -> None:
        errors.append(f"FAIL: {msg}")

    def ok(msg: str) -> None:
        print(f"  PASS: {msg}")

    print("=== ScaffoldGenerator self-test ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ScaffoldGenerator(verbose=True)

        # -----------------------------------------------------------------------
        # Test 1: Generate scaffold
        # -----------------------------------------------------------------------
        print("\n--- Test 1: Generate scaffold ---")
        try:
            gen.generate(tmpdir, project_name="test_world_model")
            ok("generate() completed without error")
        except Exception as exc:
            fail(f"generate() raised: {exc}")
            print(f"\n{len(errors)} error(s). Aborting.")
            sys.exit(1)

        # -----------------------------------------------------------------------
        # Test 2: validate_scaffold
        # -----------------------------------------------------------------------
        print("\n--- Test 2: validate_scaffold ---")
        try:
            valid = gen.validate_scaffold(tmpdir)
            if valid:
                ok("validate_scaffold() returned True")
            else:
                fail("validate_scaffold() returned False — some files missing")
        except Exception as exc:
            fail(f"validate_scaffold() raised: {exc}")

        # -----------------------------------------------------------------------
        # Test 3: All .py files are non-empty and parse as valid Python
        # -----------------------------------------------------------------------
        print("\n--- Test 3: Python file validity ---")
        pkg_dir = Path(tmpdir) / "world_model"
        py_files = list(pkg_dir.rglob("*.py"))
        if len(py_files) == 0:
            fail("No .py files found in generated scaffold")
        else:
            ok(f"Found {len(py_files)} .py files")

        for py_path in py_files:
            try:
                source = py_path.read_text(encoding="utf-8")
                if len(source.strip()) == 0:
                    fail(f"Empty .py file: {py_path}")
                    continue
                ast.parse(source)
                ok(f"Valid Python: {py_path.relative_to(tmpdir)}")
            except SyntaxError as exc:
                fail(f"Syntax error in {py_path}: {exc}")
            except Exception as exc:
                fail(f"Error reading {py_path}: {exc}")

        # -----------------------------------------------------------------------
        # Test 4: All required YAML files exist and are non-empty
        # -----------------------------------------------------------------------
        print("\n--- Test 4: YAML config files ---")
        configs_dir = pkg_dir / "configs"
        for cfg in CONFIG_FILES:
            path = configs_dir / cfg
            if path.is_file() and path.stat().st_size > 0:
                ok(f"Config file exists: {cfg}")
            else:
                fail(f"Missing or empty config file: {cfg}")

        # -----------------------------------------------------------------------
        # Test 5: pyproject.toml exists
        # -----------------------------------------------------------------------
        print("\n--- Test 5: pyproject.toml ---")
        toml_path = Path(tmpdir) / "pyproject.toml"
        if toml_path.is_file() and toml_path.stat().st_size > 0:
            ok("pyproject.toml exists and is non-empty")
        else:
            fail("pyproject.toml missing or empty")

        # -----------------------------------------------------------------------
        # Test 6: README.md exists
        # -----------------------------------------------------------------------
        print("\n--- Test 6: README.md ---")
        readme_path = Path(tmpdir) / "README.md"
        if readme_path.is_file() and readme_path.stat().st_size > 0:
            ok("README.md exists and is non-empty")
        else:
            fail("README.md missing or empty")

        # -----------------------------------------------------------------------
        # Test 7: skip_configs flag
        # -----------------------------------------------------------------------
        print("\n--- Test 7: skip_configs=True ---")
        with tempfile.TemporaryDirectory() as tmpdir2:
            gen2 = ScaffoldGenerator(verbose=False)
            gen2.generate(tmpdir2, skip_configs=True)
            for cfg in CONFIG_FILES:
                path = Path(tmpdir2) / "world_model" / "configs" / cfg
                if not path.exists():
                    ok(f"Config skipped: {cfg}")
                else:
                    fail(f"Config should be absent with skip_configs=True: {cfg}")

        # -----------------------------------------------------------------------
        # Test 8: skip_readme flag
        # -----------------------------------------------------------------------
        print("\n--- Test 8: skip_readme=True ---")
        with tempfile.TemporaryDirectory() as tmpdir3:
            gen3 = ScaffoldGenerator(verbose=False)
            gen3.generate(tmpdir3, skip_readme=True)
            readme = Path(tmpdir3) / "README.md"
            if not readme.exists():
                ok("README.md absent with skip_readme=True")
            else:
                fail("README.md should be absent with skip_readme=True")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n=== Summary ===")
    if errors:
        for e in errors:
            print(f"  {e}")
        print(f"\n{len(errors)} test(s) FAILED.")
        sys.exit(1)
    else:
        print("All ScaffoldGenerator tests PASSED.")
        sys.exit(0)
