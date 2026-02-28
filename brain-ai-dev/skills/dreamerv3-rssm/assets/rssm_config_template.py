"""
RSSM Configuration Template — DreamerV3 Dataclasses

Provides:
  RSSMConfig:          Core RSSM hyperparameters
  SymlogTwohotConfig:  Bin layout for prediction heads
  LossConfig:          KL balancing and prediction loss weights
  RSSMState:           State container (deter + stoch + logits)
  ImaginedTrajectory:  Imagination rollout output
  LossResult:          Loss decomposition
  MODEL_SIZES:         Standard DreamerV3 size presets (12M - 200M)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RSSMConfig:
    """
    Core RSSM hyperparameters.

    Attributes:
        deter_dim:   Block GRU hidden dimension (deterministic state size).
        stoch_dim:   Number of independent categorical distributions.
        num_classes: Classes per categorical distribution.
        hidden_dim:  MLP hidden width for prior/posterior networks.
        num_layers:  Number of hidden layers in prior/posterior MLPs.
        activation:  Nonlinearity name ("silu" only; others unsupported).
        norm:        Normalization type ("layernorm" for MLP, RMSNorm on GRU output).
        unimix:      Uniform mixture fraction for categorical distributions.
    """
    deter_dim: int = 1024
    stoch_dim: int = 32
    num_classes: int = 32
    hidden_dim: int = 1024
    num_layers: int = 2
    activation: str = "silu"
    norm: str = "layernorm"
    unimix: float = 0.01

    @property
    def stoch_flat_dim(self) -> int:
        """Flattened stochastic state dimension: stoch_dim * num_classes."""
        return self.stoch_dim * self.num_classes

    @property
    def feature_dim(self) -> int:
        """Full feature vector dimension: deter_dim + stoch_dim * num_classes."""
        return self.deter_dim + self.stoch_flat_dim

    def validate(self) -> None:
        """Raise ValueError if any field is out of valid range."""
        if self.deter_dim <= 0:
            raise ValueError(f"deter_dim must be > 0, got {self.deter_dim}")
        if self.stoch_dim <= 0:
            raise ValueError(f"stoch_dim must be > 0, got {self.stoch_dim}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.activation not in ("silu",):
            raise ValueError(f"activation must be 'silu', got {self.activation!r}")
        if self.norm not in ("layernorm",):
            raise ValueError(f"norm must be 'layernorm', got {self.norm!r}")
        if not (0.0 <= self.unimix < 1.0):
            raise ValueError(f"unimix must be in [0, 1), got {self.unimix}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RSSMConfig":
        """Deserialize from a dictionary (e.g., loaded from JSON)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SymlogTwohotConfig:
    """
    Configuration for symlog twohot prediction heads.

    Attributes:
        num_bins: Number of bins for the twohot distribution.
        low:      Minimum bin center in symlog space (covers symexp(-20) ≈ -485M).
        high:     Maximum bin center in symlog space (covers symexp(20) ≈ +485M).
    """
    num_bins: int = 255
    low: float = -20.0
    high: float = 20.0

    def validate(self) -> None:
        if self.num_bins < 3:
            raise ValueError(f"num_bins must be >= 3, got {self.num_bins}")
        if self.low >= self.high:
            raise ValueError(f"low must be < high, got low={self.low}, high={self.high}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SymlogTwohotConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class LossConfig:
    """
    World model loss hyperparameters.

    Attributes:
        kl_free_nats:    Free-nats threshold. KL below this value gets no gradient.
        kl_dyn_scale:    Coefficient on the dynamics KL loss (trains prior).
        kl_rep_scale:    Coefficient on the representation KL loss (trains posterior).
        reward_scale:    Coefficient on the reward prediction loss.
        continue_scale:  Coefficient on the continue prediction loss.
        obs_scale:       Coefficient on the observation reconstruction loss.
    """
    kl_free_nats: float = 1.0
    kl_dyn_scale: float = 0.5
    kl_rep_scale: float = 0.1
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    obs_scale: float = 1.0

    def validate(self) -> None:
        if self.kl_free_nats < 0:
            raise ValueError(f"kl_free_nats must be >= 0, got {self.kl_free_nats}")
        if self.kl_dyn_scale < 0:
            raise ValueError(f"kl_dyn_scale must be >= 0, got {self.kl_dyn_scale}")
        if self.kl_rep_scale < 0:
            raise ValueError(f"kl_rep_scale must be >= 0, got {self.kl_rep_scale}")
        for name in ("reward_scale", "continue_scale", "obs_scale"):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LossConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# State and Output Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RSSMState:
    """
    Container for one RSSM latent state (h_t, z_t).

    Attributes:
        deter:  Deterministic state h_t, shape (batch, deter_dim).
        stoch:  Stochastic state z_t, shape (batch, stoch_dim, num_classes).
        logits: Distribution logits used to produce z_t, shape (batch, stoch_dim, num_classes).
                For initial state, logits are zeros.
    """
    deter: Tensor
    stoch: Tensor
    logits: Tensor

    @property
    def features(self) -> Tensor:
        """
        Feature vector for prediction heads.

        Returns concat(deter, flatten(stoch)).
        Shape: (batch, deter_dim + stoch_dim * num_classes).
        """
        stoch_flat = self.stoch.flatten(start_dim=-2)  # (batch, stoch_dim * num_classes)
        return torch.cat([self.deter, stoch_flat], dim=-1)

    @property
    def batch_size(self) -> int:
        return self.deter.shape[0]

    @property
    def device(self) -> torch.device:
        return self.deter.device

    def detach(self) -> "RSSMState":
        """Return a new RSSMState with all tensors detached from the computation graph."""
        return RSSMState(
            deter=self.deter.detach(),
            stoch=self.stoch.detach(),
            logits=self.logits.detach(),
        )

    def to(self, device: torch.device) -> "RSSMState":
        """Move all tensors to a device."""
        return RSSMState(
            deter=self.deter.to(device),
            stoch=self.stoch.to(device),
            logits=self.logits.to(device),
        )


@dataclass
class ImaginedTrajectory:
    """
    Output of an imagination rollout.

    All tensors have leading dimension = horizon.

    Attributes:
        features:        Feature vectors, shape (horizon, batch, feature_dim).
        actions:         Actions taken by policy, shape (horizon, batch, action_dim).
        reward_logits:   Reward prediction logits, shape (horizon, batch, num_bins).
        continue_logits: Continue prediction logits, shape (horizon, batch, 1).
    """
    features: Tensor
    actions: Tensor
    reward_logits: Tensor
    continue_logits: Tensor

    @property
    def horizon(self) -> int:
        return self.features.shape[0]

    @property
    def batch_size(self) -> int:
        return self.features.shape[1]

    @property
    def continue_probs(self) -> Tensor:
        """Sigmoid of continue logits. Shape: (horizon, batch)."""
        return torch.sigmoid(self.continue_logits.squeeze(-1))

    def to(self, device: torch.device) -> "ImaginedTrajectory":
        return ImaginedTrajectory(
            features=self.features.to(device),
            actions=self.actions.to(device),
            reward_logits=self.reward_logits.to(device),
            continue_logits=self.continue_logits.to(device),
        )


@dataclass
class LossResult:
    """
    Decomposed world model loss.

    Attributes:
        total:        Weighted sum of all losses.
        kl_dyn:       Dynamics KL term: max(free, KL[sg(post) || prior]).
        kl_rep:       Representation KL term: max(free, KL[post || sg(prior)]).
        obs_loss:     Observation reconstruction loss.
        reward_loss:  Reward prediction loss (symlog twohot cross-entropy).
        continue_loss: Continue prediction loss (binary cross-entropy).
    """
    total: Tensor
    kl_dyn: Tensor
    kl_rep: Tensor
    obs_loss: Tensor
    reward_loss: Tensor
    continue_loss: Tensor

    def to_dict(self) -> dict[str, float]:
        """Return a dictionary of scalar loss values for logging."""
        return {
            "total": self.total.item(),
            "kl_dyn": self.kl_dyn.item(),
            "kl_rep": self.kl_rep.item(),
            "obs_loss": self.obs_loss.item(),
            "reward_loss": self.reward_loss.item(),
            "continue_loss": self.continue_loss.item(),
        }


# ---------------------------------------------------------------------------
# Model Size Presets
# ---------------------------------------------------------------------------

# Standard DreamerV3 model sizes from the official configuration files.
# Each entry overrides specific fields of RSSMConfig defaults.
MODEL_SIZES: dict[str, dict[str, Any]] = {
    "12M": {
        "deter_dim": 512,
        "hidden_dim": 256,
        "num_classes": 16,
        "stoch_dim": 32,
    },
    "25M": {
        "deter_dim": 1024,
        "hidden_dim": 384,
        "num_classes": 24,
        "stoch_dim": 32,
    },
    "50M": {
        "deter_dim": 2048,
        "hidden_dim": 512,
        "num_classes": 32,
        "stoch_dim": 32,
    },
    "100M": {
        "deter_dim": 3072,
        "hidden_dim": 768,
        "num_classes": 48,
        "stoch_dim": 32,
    },
    "200M": {
        "deter_dim": 4096,
        "hidden_dim": 1024,
        "num_classes": 64,
        "stoch_dim": 32,
    },
}


def get_config_for_size(size: str) -> RSSMConfig:
    """
    Return a validated RSSMConfig for a given model size string.

    Args:
        size: One of "12M", "25M", "50M", "100M", "200M".

    Returns:
        RSSMConfig configured for the specified size.

    Raises:
        KeyError: If size is not a recognized model size.
    """
    if size not in MODEL_SIZES:
        raise KeyError(
            f"Unknown model size {size!r}. "
            f"Available: {list(MODEL_SIZES.keys())}"
        )
    overrides = MODEL_SIZES[size]
    # Build cleanly using dataclass defaults + overrides
    base = RSSMConfig()
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    base.validate()
    return base


# ---------------------------------------------------------------------------
# JSON Serialization Helpers
# ---------------------------------------------------------------------------

def configs_to_json(
    rssm_cfg: RSSMConfig,
    twohot_cfg: SymlogTwohotConfig,
    loss_cfg: LossConfig,
) -> str:
    """Serialize all three configs to a JSON string."""
    return json.dumps(
        {
            "rssm": rssm_cfg.to_dict(),
            "twohot": twohot_cfg.to_dict(),
            "loss": loss_cfg.to_dict(),
        },
        indent=2,
    )


def configs_from_json(json_str: str) -> tuple[RSSMConfig, SymlogTwohotConfig, LossConfig]:
    """Deserialize all three configs from a JSON string."""
    data = json.loads(json_str)
    return (
        RSSMConfig.from_dict(data.get("rssm", {})),
        SymlogTwohotConfig.from_dict(data.get("twohot", {})),
        LossConfig.from_dict(data.get("loss", {})),
    )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("RSSM Config self-tests")
    print("=" * 60)

    FAILURES: list[str] = []

    def check(condition: bool, name: str, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        if not condition:
            FAILURES.append(name)

    # -----------------------------------------------------------------------
    # Test 1: Default configs are valid
    # -----------------------------------------------------------------------
    print("Test 1: Default configs valid")

    try:
        rc = RSSMConfig()
        rc.validate()
        check(True, "RSSMConfig default validates")
    except Exception as e:
        check(False, "RSSMConfig default validates", str(e))

    try:
        tc = SymlogTwohotConfig()
        tc.validate()
        check(True, "SymlogTwohotConfig default validates")
    except Exception as e:
        check(False, "SymlogTwohotConfig default validates", str(e))

    try:
        lc = LossConfig()
        lc.validate()
        check(True, "LossConfig default validates")
    except Exception as e:
        check(False, "LossConfig default validates", str(e))

    # -----------------------------------------------------------------------
    # Test 2: Default field values
    # -----------------------------------------------------------------------
    print("\nTest 2: Default field values")

    rc = RSSMConfig()
    check(rc.deter_dim == 1024, f"deter_dim=1024", f"got {rc.deter_dim}")
    check(rc.stoch_dim == 32, f"stoch_dim=32", f"got {rc.stoch_dim}")
    check(rc.num_classes == 32, f"num_classes=32", f"got {rc.num_classes}")
    check(rc.hidden_dim == 1024, f"hidden_dim=1024", f"got {rc.hidden_dim}")
    check(rc.unimix == 0.01, f"unimix=0.01", f"got {rc.unimix}")
    check(rc.stoch_flat_dim == 1024, "stoch_flat_dim=1024", f"got {rc.stoch_flat_dim}")
    check(rc.feature_dim == 2048, "feature_dim=2048", f"got {rc.feature_dim}")

    lc = LossConfig()
    check(lc.kl_free_nats == 1.0, "kl_free_nats=1.0")
    check(lc.kl_dyn_scale == 0.5, "kl_dyn_scale=0.5")
    check(lc.kl_rep_scale == 0.1, "kl_rep_scale=0.1")

    tc = SymlogTwohotConfig()
    check(tc.num_bins == 255, "num_bins=255")
    check(tc.low == -20.0, "low=-20.0")
    check(tc.high == 20.0, "high=20.0")

    # -----------------------------------------------------------------------
    # Test 3: Model size presets
    # -----------------------------------------------------------------------
    print("\nTest 3: Model size presets")

    for size in MODEL_SIZES:
        try:
            cfg = get_config_for_size(size)
            cfg.validate()
            check(True, f"size {size} validates")
            overrides = MODEL_SIZES[size]
            for k, v in overrides.items():
                actual = getattr(cfg, k)
                check(actual == v, f"size {size}: {k}={v}", f"got {actual}")
        except Exception as e:
            check(False, f"size {size} validates", str(e))

    # -----------------------------------------------------------------------
    # Test 4: Serialization round-trip
    # -----------------------------------------------------------------------
    print("\nTest 4: Serialization round-trip")

    rc_orig = RSSMConfig(deter_dim=2048, stoch_dim=16, num_classes=16, hidden_dim=512)
    tc_orig = SymlogTwohotConfig(num_bins=127, low=-15.0, high=15.0)
    lc_orig = LossConfig(kl_free_nats=2.0, kl_dyn_scale=0.8, kl_rep_scale=0.2)

    json_str = configs_to_json(rc_orig, tc_orig, lc_orig)
    rc_rt, tc_rt, lc_rt = configs_from_json(json_str)

    check(rc_rt.deter_dim == 2048, "rssm round-trip deter_dim")
    check(rc_rt.stoch_dim == 16, "rssm round-trip stoch_dim")
    check(tc_rt.num_bins == 127, "twohot round-trip num_bins")
    check(tc_rt.low == -15.0, "twohot round-trip low")
    check(lc_rt.kl_free_nats == 2.0, "loss round-trip kl_free_nats")
    check(lc_rt.kl_dyn_scale == 0.8, "loss round-trip kl_dyn_scale")

    # to_dict / from_dict for each class
    rc_d = rc_orig.to_dict()
    rc_recovered = RSSMConfig.from_dict(rc_d)
    check(rc_recovered.deter_dim == rc_orig.deter_dim, "RSSMConfig.from_dict round-trip")

    lc_d = lc_orig.to_dict()
    lc_recovered = LossConfig.from_dict(lc_d)
    check(lc_recovered.kl_free_nats == lc_orig.kl_free_nats, "LossConfig.from_dict round-trip")

    # -----------------------------------------------------------------------
    # Test 5: Validation rejects invalid values
    # -----------------------------------------------------------------------
    print("\nTest 5: Validation rejects invalid values")

    invalid_cases = [
        (lambda: RSSMConfig(deter_dim=0).validate(), "deter_dim=0"),
        (lambda: RSSMConfig(stoch_dim=-1).validate(), "stoch_dim=-1"),
        (lambda: RSSMConfig(num_classes=1).validate(), "num_classes=1"),
        (lambda: RSSMConfig(unimix=1.5).validate(), "unimix=1.5"),
        (lambda: RSSMConfig(activation="relu").validate(), "activation=relu"),
        (lambda: SymlogTwohotConfig(num_bins=2).validate(), "num_bins=2"),
        (lambda: SymlogTwohotConfig(low=10.0, high=5.0).validate(), "low > high"),
        (lambda: LossConfig(kl_free_nats=-1.0).validate(), "kl_free_nats=-1"),
        (lambda: LossConfig(reward_scale=-0.5).validate(), "reward_scale=-0.5"),
    ]

    for fn, desc in invalid_cases:
        try:
            fn()
            check(False, f"should reject {desc}", "no exception raised")
        except ValueError:
            check(True, f"rejects invalid {desc}")

    # -----------------------------------------------------------------------
    # Test 6: RSSMState features property
    # -----------------------------------------------------------------------
    print("\nTest 6: RSSMState features property")

    deter = torch.zeros(4, 64)
    stoch = torch.zeros(4, 8, 8)
    logits = torch.zeros(4, 8, 8)
    state = RSSMState(deter=deter, stoch=stoch, logits=logits)
    feat = state.features
    check(feat.shape == (4, 64 + 8 * 8), "features shape", str(feat.shape))
    check(state.batch_size == 4, "batch_size property")

    detached = state.detach()
    check(not detached.deter.requires_grad, "detach: deter not requires_grad")

    # -----------------------------------------------------------------------
    # Test 7: ImaginedTrajectory properties
    # -----------------------------------------------------------------------
    print("\nTest 7: ImaginedTrajectory properties")

    traj = ImaginedTrajectory(
        features=torch.zeros(15, 4, 128),
        actions=torch.zeros(15, 4, 2),
        reward_logits=torch.zeros(15, 4, 255),
        continue_logits=torch.zeros(15, 4, 1),
    )
    check(traj.horizon == 15, "horizon=15")
    check(traj.batch_size == 4, "batch_size=4")
    check(traj.continue_probs.shape == (15, 4), "continue_probs shape")

    # -----------------------------------------------------------------------
    # Test 8: LossResult to_dict
    # -----------------------------------------------------------------------
    print("\nTest 8: LossResult.to_dict")

    lr = LossResult(
        total=torch.tensor(5.0),
        kl_dyn=torch.tensor(2.0),
        kl_rep=torch.tensor(0.5),
        obs_loss=torch.tensor(1.5),
        reward_loss=torch.tensor(0.8),
        continue_loss=torch.tensor(0.2),
    )
    d = lr.to_dict()
    check(isinstance(d, dict), "to_dict returns dict")
    check(abs(d["total"] - 5.0) < 1e-6, "to_dict total correct")
    check("kl_dyn" in d, "to_dict has kl_dyn")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} test(s): {', '.join(FAILURES)}")
        sys.exit(1)
    else:
        print("All tests PASSED.")
