"""
SearchSpace — Declarative hyperparameter search space definition.

Provides parameter types (float, int, categorical, conditional), sampling
(uniform, log-uniform), grid enumeration, and config-class introspection.

No external dependencies beyond Python stdlib + random/math.

Usage:
    from search_space_template import SearchSpace

    space = SearchSpace()
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_int("num_layers", 2, 12)
    space.add_categorical("optimizer", ["adam", "sgd", "adamw"])
    space.add_conditional("beta1", "optimizer == adam", sub)

    sample = space.sample()   # -> {"lr": 0.003, "num_layers": 7, "optimizer": "sgd"}
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Parameter descriptors
# ---------------------------------------------------------------------------

@dataclass
class FloatParam:
    """Continuous parameter on [low, high], optionally log-scaled."""
    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"FloatParam '{self.name}': low ({self.low}) must be < high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError(f"FloatParam '{self.name}': log-scale requires low > 0, got {self.low}")

    def sample(self, rng: random.Random) -> float:
        if self.log:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return math.exp(rng.uniform(log_low, log_high))
        return rng.uniform(self.low, self.high)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def grid(self, n: int) -> List[float]:
        """Return *n* evenly-spaced grid points (log-spaced if log=True)."""
        if n < 1:
            raise ValueError("grid size must be >= 1")
        if n == 1:
            return [(self.low + self.high) / 2.0]
        if self.log:
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return [math.exp(log_low + i * (log_high - log_low) / (n - 1)) for i in range(n)]
        return [self.low + i * (self.high - self.low) / (n - 1) for i in range(n)]


@dataclass
class IntParam:
    """Integer parameter on [low, high] inclusive."""
    name: str
    low: int
    high: int

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"IntParam '{self.name}': low ({self.low}) must be <= high ({self.high})")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)

    def contains(self, value: int) -> bool:
        return self.low <= value <= self.high and isinstance(value, int)

    def grid(self, n: Optional[int] = None) -> List[int]:
        """Return grid; if *n* is None, return all integers in [low, high]."""
        all_vals = list(range(self.low, self.high + 1))
        if n is None or n >= len(all_vals):
            return all_vals
        if n < 1:
            raise ValueError("grid size must be >= 1")
        if n == 1:
            return [(self.low + self.high) // 2]
        step = max(1, (self.high - self.low) / (n - 1))
        return sorted(set(int(round(self.low + i * step)) for i in range(n)))


@dataclass
class CategoricalParam:
    """Categorical (discrete) parameter."""
    name: str
    choices: List[Any]

    def __post_init__(self):
        if not self.choices:
            raise ValueError(f"CategoricalParam '{self.name}': choices must be non-empty")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.choices)

    def contains(self, value: Any) -> bool:
        return value in self.choices

    def grid(self, n: Optional[int] = None) -> List[Any]:
        if n is None or n >= len(self.choices):
            return list(self.choices)
        return list(self.choices[:n])


@dataclass
class ConditionalParam:
    """A parameter or sub-space that is active only when *condition* is met.

    ``condition`` is a simple string expression evaluated against the parent
    sample, e.g. ``"optimizer == adam"`` or ``"use_htm == True"``.
    """
    name: str
    condition: str
    sub_space: "SearchSpace"

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_condition(condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple ``key == value`` or ``key != value`` condition."""
        condition = condition.strip()
        for op, negate in [("!=", True), ("==", False)]:
            if op in condition:
                lhs, rhs = [s.strip() for s in condition.split(op, 1)]
                if lhs not in context:
                    return False
                lhs_val = context[lhs]
                # Coerce rhs to the type of lhs_val
                rhs_val = _coerce(rhs, type(lhs_val))
                result = lhs_val == rhs_val
                return (not result) if negate else result
        raise ValueError(f"Cannot parse condition: '{condition}'")


ParamType = Union[FloatParam, IntParam, CategoricalParam]


def _coerce(s: str, target_type: type) -> Any:
    """Best-effort coerce a string *s* to *target_type*."""
    if target_type is bool:
        return s.strip().lower() in ("true", "1", "yes")
    if target_type is int:
        return int(s)
    if target_type is float:
        return float(s)
    return s  # str stays str


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------

class SearchSpace:
    """Declarative search space.

    Supports float (uniform / log-uniform), int, categorical, and conditional
    parameters.  Provides ``sample()``, ``grid()``, ``size()``, and
    ``from_config_class()`` for BrainAIConfig introspection.
    """

    def __init__(self, seed: Optional[int] = None):
        self._params: Dict[str, ParamType] = {}
        self._conditionals: List[ConditionalParam] = []
        self._order: List[str] = []  # insertion order for param names
        self._rng = random.Random(seed)

    # -- builder API --------------------------------------------------------

    def add_float(self, name: str, low: float, high: float,
                  log: bool = False) -> "SearchSpace":
        p = FloatParam(name=name, low=low, high=high, log=log)
        self._params[name] = p
        if name not in self._order:
            self._order.append(name)
        return self

    def add_int(self, name: str, low: int, high: int) -> "SearchSpace":
        p = IntParam(name=name, low=low, high=high)
        self._params[name] = p
        if name not in self._order:
            self._order.append(name)
        return self

    def add_categorical(self, name: str, choices: List[Any]) -> "SearchSpace":
        p = CategoricalParam(name=name, choices=list(choices))
        self._params[name] = p
        if name not in self._order:
            self._order.append(name)
        return self

    def add_conditional(self, name: str, condition: str,
                        sub_space: "SearchSpace") -> "SearchSpace":
        self._conditionals.append(
            ConditionalParam(name=name, condition=condition, sub_space=sub_space)
        )
        return self

    # -- sampling -----------------------------------------------------------

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Sample a single configuration from this space."""
        r = rng or self._rng
        result: Dict[str, Any] = {}
        for name in self._order:
            param = self._params[name]
            result[name] = param.sample(r)
        # Evaluate conditionals
        for cond in self._conditionals:
            if ConditionalParam._eval_condition(cond.condition, result):
                sub_sample = cond.sub_space.sample(r)
                result.update(sub_sample)
        return result

    def sample_n(self, n: int, rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
        """Sample *n* configurations."""
        return [self.sample(rng) for _ in range(n)]

    # -- grid ---------------------------------------------------------------

    def grid(self, resolution: Optional[Dict[str, int]] = None,
             default_n: int = 5) -> List[Dict[str, Any]]:
        """Return a full grid (Cartesian product) of parameter values.

        Parameters
        ----------
        resolution : per-parameter grid size (overrides *default_n*)
        default_n : default number of grid points per param
        """
        if not self._params:
            return [{}]
        resolution = resolution or {}
        grids: Dict[str, List[Any]] = {}
        for name in self._order:
            param = self._params[name]
            n = resolution.get(name, default_n)
            grids[name] = param.grid(n)
        keys = list(self._order)
        combos = list(itertools.product(*(grids[k] for k in keys)))
        return [dict(zip(keys, vals)) for vals in combos]

    def grid_size(self, resolution: Optional[Dict[str, int]] = None,
                  default_n: int = 5) -> int:
        """Number of points in the grid without materializing it."""
        resolution = resolution or {}
        total = 1
        for name in self._order:
            param = self._params[name]
            n = resolution.get(name, default_n)
            total *= len(param.grid(n))
        return total

    # -- properties ---------------------------------------------------------

    @property
    def param_names(self) -> List[str]:
        return list(self._order)

    @property
    def params(self) -> Dict[str, ParamType]:
        return dict(self._params)

    @property
    def conditionals(self) -> List[ConditionalParam]:
        return list(self._conditionals)

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __repr__(self) -> str:
        parts = [f"SearchSpace({len(self._params)} params"]
        if self._conditionals:
            parts.append(f", {len(self._conditionals)} conditional")
        parts.append(")")
        return "".join(parts)

    # -- introspection from dataclass config --------------------------------

    @classmethod
    def from_config_class(cls, config_cls: type,
                          overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                          seed: Optional[int] = None) -> "SearchSpace":
        """Build a SearchSpace by inspecting a ``@dataclass`` configuration class.

        For each field, a default range is inferred:
          - float  -> [default*0.1, default*10] (log-scale if default < 1)
          - int    -> [max(1, default//4), default*4]
          - bool   -> [True, False]
          - str    -> skipped (need choices in overrides)

        Parameters
        ----------
        config_cls : A dataclass type.
        overrides  : Dict mapping field names to override dicts, e.g.
                     ``{"beta": {"low": 0.9, "high": 0.99}}``
        """
        space = cls(seed=seed)
        overrides = overrides or {}
        for f in dc_fields(config_cls):
            name = f.name
            ovr = overrides.get(name, {})
            default_val = f.default if f.default is not f.default_factory else None  # type: ignore[attr-defined]
            # try to get default
            try:
                default_val = f.default
                if default_val is getattr(f, "default_factory", None):
                    default_val = None
            except Exception:
                default_val = None

            # Determine type
            ftype = f.type if isinstance(f.type, type) else None
            if ftype is None:
                # try string annotation
                ftype_str = str(f.type).lower()
                if "float" in ftype_str:
                    ftype = float
                elif "int" in ftype_str:
                    ftype = int
                elif "bool" in ftype_str:
                    ftype = bool
                elif "str" in ftype_str:
                    ftype = str

            if ftype is float:
                low = ovr.get("low", max(1e-10, default_val * 0.1) if default_val and default_val > 0 else 1e-6)
                high = ovr.get("high", default_val * 10.0 if default_val and default_val > 0 else 1.0)
                log = ovr.get("log", default_val is not None and 0 < default_val < 1.0)
                space.add_float(name, low, high, log=log)
            elif ftype is int:
                if default_val is not None and isinstance(default_val, int) and default_val > 0:
                    low = ovr.get("low", max(1, default_val // 4))
                    high = ovr.get("high", default_val * 4)
                else:
                    low = ovr.get("low", 1)
                    high = ovr.get("high", 10)
                space.add_int(name, low, high)
            elif ftype is bool:
                space.add_categorical(name, ovr.get("choices", [True, False]))
            elif ftype is str:
                if "choices" in ovr:
                    space.add_categorical(name, ovr["choices"])
                # else: skip (cannot infer string choices)
        return space

    # -- from preset --------------------------------------------------------

    @classmethod
    def from_preset(cls, preset_params: List[Dict[str, Any]],
                    seed: Optional[int] = None) -> "SearchSpace":
        """Build a SearchSpace from a PhasePresets parameter list.

        Each dict has: name, type, low, high, log, choices, condition.
        """
        space = cls(seed=seed)
        conditional_buffer: Dict[str, List[Dict[str, Any]]] = {}
        unconditional = []

        for p in preset_params:
            if p.get("condition"):
                cond_key = p["condition"]
                conditional_buffer.setdefault(cond_key, []).append(p)
            else:
                unconditional.append(p)

        # Add unconditional params
        for p in unconditional:
            _add_param(space, p)

        # Add conditional groups
        for cond_str, params in conditional_buffer.items():
            sub = cls(seed=seed)
            for p in params:
                _add_param(sub, p)
            cond_name = f"cond_{cond_str.replace(' ', '_')}"
            space.add_conditional(cond_name, cond_str, sub)

        return space

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the search space to a plain dict."""
        params = []
        for name in self._order:
            p = self._params[name]
            if isinstance(p, FloatParam):
                params.append({"name": name, "type": "float",
                               "low": p.low, "high": p.high, "log": p.log})
            elif isinstance(p, IntParam):
                params.append({"name": name, "type": "int",
                               "low": p.low, "high": p.high})
            elif isinstance(p, CategoricalParam):
                params.append({"name": name, "type": "categorical",
                               "choices": p.choices})
        conditionals = []
        for c in self._conditionals:
            conditionals.append({
                "name": c.name, "condition": c.condition,
                "sub_space": c.sub_space.to_dict()
            })
        return {"params": params, "conditionals": conditionals}

    @classmethod
    def from_dict(cls, d: Dict[str, Any],
                  seed: Optional[int] = None) -> "SearchSpace":
        """Deserialize from a dict."""
        space = cls(seed=seed)
        for p in d.get("params", []):
            if p["type"] == "float":
                space.add_float(p["name"], p["low"], p["high"], p.get("log", False))
            elif p["type"] == "int":
                space.add_int(p["name"], p["low"], p["high"])
            elif p["type"] == "categorical":
                space.add_categorical(p["name"], p["choices"])
        for c in d.get("conditionals", []):
            sub = cls.from_dict(c["sub_space"], seed=seed)
            space.add_conditional(c["name"], c["condition"], sub)
        return space

    # -- merge / copy -------------------------------------------------------

    def copy(self) -> "SearchSpace":
        return SearchSpace.from_dict(self.to_dict())

    def merge(self, other: "SearchSpace") -> "SearchSpace":
        """Merge another space into this one (mutates self)."""
        for name in other._order:
            self._params[name] = copy.deepcopy(other._params[name])
            if name not in self._order:
                self._order.append(name)
        for c in other._conditionals:
            self._conditionals.append(copy.deepcopy(c))
        return self


def _add_param(space: SearchSpace, p: Dict[str, Any]) -> None:
    """Helper: add a single param dict to a space."""
    if p["type"] == "float":
        space.add_float(p["name"], p["low"], p["high"], p.get("log", False))
    elif p["type"] == "int":
        space.add_int(p["name"], p["low"], p["high"])
    elif p["type"] == "categorical":
        space.add_categorical(p["name"], p["choices"])


# ---------------------------------------------------------------------------
# Self-tests  (30+ tests)
# ---------------------------------------------------------------------------

def _run_tests():
    passed = 0
    failed = 0

    def check(name: str, condition: bool, msg: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — {msg}")

    rng = random.Random(42)

    print("=== FloatParam Tests ===")

    # SP-01: Float parameter
    fp = FloatParam("x", 0.0, 1.0)
    vals = [fp.sample(rng) for _ in range(1000)]
    check("SP-01 float in bounds", all(0.0 <= v <= 1.0 for v in vals))

    # SP-02: Float log-scale
    fp_log = FloatParam("lr", 1e-5, 1e-1, log=True)
    vals_log = [fp_log.sample(rng) for _ in range(10000)]
    check("SP-02 log-float in bounds", all(1e-5 <= v <= 1e-1 for v in vals_log))
    # Check that median is near geometric mean, not arithmetic mean
    geo_mean = math.sqrt(1e-5 * 1e-1)  # ~3.16e-3
    arith_mean = (1e-5 + 1e-1) / 2  # ~0.05
    median_val = sorted(vals_log)[5000]
    check("SP-02 log distribution", abs(math.log10(median_val) - math.log10(geo_mean)) < 0.5,
          f"median={median_val:.4e}, geo_mean={geo_mean:.4e}")

    # Float grid
    grid = fp.grid(5)
    check("SP-01b float grid len", len(grid) == 5)
    check("SP-01b float grid range", abs(grid[0] - 0.0) < 1e-9 and abs(grid[-1] - 1.0) < 1e-9)

    # Float log grid
    grid_log = fp_log.grid(3)
    check("SP-02b log grid len", len(grid_log) == 3)
    check("SP-02b log grid endpoints", abs(grid_log[0] - 1e-5) < 1e-10 and abs(grid_log[-1] - 1e-1) < 1e-6)

    # Float validation
    try:
        FloatParam("bad", 5.0, 1.0)
        check("SP-01c float low>=high", False, "should raise")
    except ValueError:
        check("SP-01c float low>=high", True)

    try:
        FloatParam("bad", -1.0, 1.0, log=True)
        check("SP-02c log negative low", False, "should raise")
    except ValueError:
        check("SP-02c log negative low", True)

    print("\n=== IntParam Tests ===")

    # SP-03: Integer parameter
    ip = IntParam("n", 1, 10)
    ivals = [ip.sample(rng) for _ in range(1000)]
    check("SP-03 int in bounds", all(1 <= v <= 10 for v in ivals))
    check("SP-03 int type", all(isinstance(v, int) for v in ivals))

    # Int grid
    igrid = ip.grid()
    check("SP-03b int grid complete", igrid == list(range(1, 11)))
    igrid3 = ip.grid(3)
    check("SP-03c int grid partial", len(igrid3) >= 3)

    # Int validation
    try:
        IntParam("bad", 10, 5)
        check("SP-03d int low>high", False, "should raise")
    except ValueError:
        check("SP-03d int low>high", True)

    print("\n=== CategoricalParam Tests ===")

    # SP-04: Categorical parameter
    cp = CategoricalParam("opt", ["adam", "sgd", "adamw"])
    cvals = [cp.sample(rng) for _ in range(1000)]
    check("SP-04 categorical in choices", all(v in ["adam", "sgd", "adamw"] for v in cvals))
    # Check all choices appear
    check("SP-04 all choices sampled", set(cvals) == {"adam", "sgd", "adamw"})

    # SP-05: Boolean via categorical
    bp = CategoricalParam("flag", [True, False])
    bvals = [bp.sample(rng) for _ in range(1000)]
    check("SP-05 bool values", all(isinstance(v, bool) for v in bvals))
    check("SP-05 both values", True in bvals and False in bvals)

    # Grid
    cgrid = cp.grid()
    check("SP-04b cat grid", cgrid == ["adam", "sgd", "adamw"])

    # Validation
    try:
        CategoricalParam("bad", [])
        check("SP-04c empty choices", False, "should raise")
    except ValueError:
        check("SP-04c empty choices", True)

    print("\n=== SearchSpace Tests ===")

    # SP-06: Sample returns complete dict
    space = SearchSpace(seed=42)
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_int("layers", 2, 12)
    space.add_float("dropout", 0.0, 0.5)
    space.add_categorical("opt", ["adam", "sgd"])
    space.add_int("batch_size", 8, 64)
    s = space.sample()
    check("SP-06 complete dict", len(s) == 5 and set(s.keys()) == {"lr", "layers", "dropout", "opt", "batch_size"})

    # SP-07: 1000 float samples in bounds
    samples = space.sample_n(1000)
    check("SP-07 float bounds", all(1e-5 <= s["lr"] <= 1e-1 for s in samples))

    # SP-08: 1000 int samples in bounds
    check("SP-08 int bounds", all(2 <= s["layers"] <= 12 and isinstance(s["layers"], int) for s in samples))

    # SP-09: 1000 categorical samples valid
    check("SP-09 categorical valid", all(s["opt"] in ["adam", "sgd"] for s in samples))

    # SP-10: Log-scale distribution
    lr_vals = [s["lr"] for s in samples]
    lr_median = sorted(lr_vals)[500]
    geo = math.sqrt(1e-5 * 1e-1)
    check("SP-10 log median",
          abs(math.log10(lr_median) - math.log10(geo)) < 0.5,
          f"median={lr_median:.4e}")

    # SP-11: Uniform distribution mean
    dropout_vals = [s["dropout"] for s in samples]
    mean_do = sum(dropout_vals) / len(dropout_vals)
    check("SP-11 uniform mean", abs(mean_do - 0.25) < 0.05, f"mean={mean_do:.3f}")

    print("\n=== Conditional Tests ===")

    # SP-12 / SP-13: Conditional active/inactive
    cond_space = SearchSpace(seed=42)
    cond_space.add_categorical("use_htm", [True, False])
    sub = SearchSpace(seed=42)
    sub.add_int("column_count", 512, 4096)
    cond_space.add_conditional("htm_params", "use_htm == True", sub)

    active_count = 0
    inactive_count = 0
    for _ in range(500):
        s = cond_space.sample()
        if s["use_htm"] is True:
            check_ok = "column_count" in s
            if check_ok:
                active_count += 1
        else:
            check_ok = "column_count" not in s
            if check_ok:
                inactive_count += 1
    check("SP-12 conditional active", active_count > 50, f"active_count={active_count}")
    check("SP-13 conditional inactive", inactive_count > 50, f"inactive_count={inactive_count}")

    # SP-14: Nested conditionals
    nested = SearchSpace(seed=42)
    nested.add_categorical("use_htm", [True, False])
    inner = SearchSpace(seed=42)
    inner.add_categorical("use_reflex", [True, False])
    reflex_sub = SearchSpace(seed=42)
    reflex_sub.add_int("reflex_tables", 4, 16)
    inner.add_conditional("reflex_params", "use_reflex == True", reflex_sub)
    inner.add_int("column_count", 512, 4096)
    nested.add_conditional("htm", "use_htm == True", inner)

    found_reflex = False
    for _ in range(500):
        s = nested.sample()
        if s.get("use_htm") and s.get("use_reflex"):
            if "reflex_tables" in s:
                found_reflex = True
                break
    check("SP-14 nested conditionals", found_reflex)

    # SP-15: Multiple conditions
    multi = SearchSpace(seed=42)
    multi.add_categorical("use_a", [True, False])
    multi.add_categorical("use_b", [True, False])
    sub_a = SearchSpace(seed=42)
    sub_a.add_float("a_param", 0.0, 1.0)
    sub_b = SearchSpace(seed=42)
    sub_b.add_float("b_param", 0.0, 1.0)
    multi.add_conditional("a_cond", "use_a == True", sub_a)
    multi.add_conditional("b_cond", "use_b == True", sub_b)
    found_both = False
    found_neither = False
    for _ in range(200):
        s = multi.sample()
        if "a_param" in s and "b_param" in s:
            found_both = True
        if "a_param" not in s and "b_param" not in s:
            found_neither = True
    check("SP-15 multiple conditions both", found_both)
    check("SP-15 multiple conditions neither", found_neither)

    print("\n=== Config Integration Tests ===")

    # SP-16 / SP-17 / SP-18: from_config_class
    @dataclass
    class DummyConfig:
        learning_rate: float = 0.001
        num_layers: int = 12
        use_dropout: bool = True
        name: str = "default"
        items: list = field(default_factory=list)

    sp = SearchSpace.from_config_class(DummyConfig)
    check("SP-16 from_config_class", "learning_rate" in sp and "num_layers" in sp)
    check("SP-18 skips non-tunable", "items" not in sp)

    sp_ovr = SearchSpace.from_config_class(
        DummyConfig,
        overrides={"learning_rate": {"low": 1e-6, "high": 1e-2, "log": True},
                   "name": {"choices": ["a", "b"]}},
    )
    check("SP-17 overrides applied", "name" in sp_ovr)
    s_ovr = sp_ovr.sample()
    check("SP-17b override range", 1e-6 <= s_ovr["learning_rate"] <= 1e-2)

    # SP-19: Empty space
    empty = SearchSpace()
    check("SP-19 empty sample", empty.sample() == {})

    # SP-20: Large space
    large = SearchSpace(seed=42)
    for i in range(50):
        large.add_float(f"p{i}", 0.0, 1.0)
    ls = large.sample()
    check("SP-20 large space", len(ls) == 50)

    print("\n=== Grid Tests ===")

    g_space = SearchSpace()
    g_space.add_categorical("opt", ["adam", "sgd"])
    g_space.add_int("n", 1, 3)
    grid = g_space.grid({"opt": 2, "n": 3}, default_n=3)
    check("Grid size", len(grid) == 6, f"got {len(grid)}")
    check("Grid size calc", g_space.grid_size({"opt": 2, "n": 3}, default_n=3) == 6)

    print("\n=== Serialization Tests ===")

    sp_ser = SearchSpace(seed=42)
    sp_ser.add_float("x", 0.0, 1.0)
    sp_ser.add_int("n", 1, 10)
    sp_ser.add_categorical("c", ["a", "b"])
    d = sp_ser.to_dict()
    sp_back = SearchSpace.from_dict(d)
    check("Serialization round-trip",
          sp_back.param_names == sp_ser.param_names)

    print("\n=== Merge / Copy Tests ===")

    sp_a = SearchSpace(seed=1)
    sp_a.add_float("x", 0, 1)
    sp_b = SearchSpace(seed=2)
    sp_b.add_int("n", 1, 5)
    sp_a.merge(sp_b)
    check("Merge", "x" in sp_a and "n" in sp_a and len(sp_a) == 2)

    sp_copy = sp_a.copy()
    check("Copy", sp_copy.param_names == sp_a.param_names)

    print("\n=== from_preset Tests ===")
    preset = [
        {"name": "lr", "type": "float", "low": 1e-5, "high": 1e-1, "log": True,
         "choices": None, "condition": None},
        {"name": "opt", "type": "categorical", "low": None, "high": None, "log": False,
         "choices": ["adam", "sgd"], "condition": None},
        {"name": "htc", "type": "int", "low": 512, "high": 4096, "log": False,
         "choices": None, "condition": "use_htm == True"},
    ]
    sp_pre = SearchSpace.from_preset(preset, seed=42)
    check("from_preset params", "lr" in sp_pre and "opt" in sp_pre)
    check("from_preset conditionals", len(sp_pre.conditionals) > 0)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
