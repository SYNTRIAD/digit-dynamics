# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research � MIT License
"""
ENGINE vNext — Module M0: Canonical Semantics & Reproducibility Backbone
Version 2.0

Architecture:
  Layer A (Semantic) — Pure data: OperationSpec, PipelineSpec, DomainPolicy
  Layer B (Execution) — Implementations: OperationExecutor, PipelineRunner

Layer A is inspectable as data (for symbolic analysis, conjecture mining).
Layer B executes. They are strictly separated.

Provides:
  1. Canonical Operation Registry (22 ops, semantic_class, structured digit_length)
  2. Canonical Pipeline DSL (string -> canonical JSON -> SHA-256)
  3. Domain Policy Specification (explicit leading-zero, repdigit, cycle policy)
  4. Deterministic Result Hashing + JSON export (op_registry_hash, fixed precision)

Usage:
    from pipeline_dsl import OperationRegistry, Pipeline, DomainPolicy, RunResult

    reg = OperationRegistry()
    pipe = Pipeline.parse("kaprekar_step |> digit_pow4 |> digit_sum", registry=reg)
    domain = DomainPolicy(base=10, digit_length=6, exclude_repdigits=True)
    result = RunResult.from_exhaustive(pipe, domain, reg)
    print(result.sha256)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Any, Set


# =============================================================================
# LAYER A — SEMANTIC LAYER (pure data, no execution logic)
# =============================================================================

# ── Enums ──

class SemanticClass(Enum):
    """Structural classification of a digit operation."""
    PERMUTATION = "permutation"       # Reorders digits (rev, sort, rotate, swap)
    DIGITWISE_MAP = "digitwise_map"   # Applies f to each digit independently (comp)
    AGGREGATE = "aggregate"           # Reduces digits to single value (ds, dp, dgcd, dxor)
    SUBTRACTIVE = "subtractive"       # Based on subtraction of digit arrangements (kap)
    COMPLEMENT = "complement"         # 9's complement family (comp)
    MIXED = "mixed"                   # Combines multiple classes (truc_1089, add_reverse)
    ARITHMETIC = "arithmetic"         # General arithmetic (collatz)


class DSClass(Enum):
    """Digit-sum behavior classification (P/C/X from paper)."""
    PRESERVING = "P"
    CONTRACTIVE = "C"
    EXPANSIVE = "X"


class LeadingZeroPolicy(Enum):
    """How leading zeros are handled."""
    DROPS = "drops"           # Leading zeros removed (int conversion)
    PADS = "pads"             # Zero-padding to maintain digit count
    PRESERVES = "preserves"   # Digit count always preserved (e.g., sort_desc)
    NA = "n/a"                # Not applicable (output is small number)


# ── Structured digit-length behavior ──

@dataclass(frozen=True)
class DigitLengthSpec:
    """Machine-readable specification of how an operation affects digit count."""
    fixed: bool = False           # Always preserves digit count
    may_reduce: bool = False      # Can reduce digit count (leading zero drop)
    may_increase: bool = False    # Can increase digit count (e.g., add_reverse)
    reduces_to_small: bool = False  # Output typically 1-2 digits (e.g., digit_sum)
    pad_internal: bool = False    # Uses zero-padding internally (e.g., kaprekar)
    variable: bool = False        # Unpredictable (e.g., collatz)

    def canonical_dict(self) -> dict:
        return {
            "fixed": self.fixed,
            "may_reduce": self.may_reduce,
            "may_increase": self.may_increase,
            "reduces_to_small": self.reduces_to_small,
            "pad_internal": self.pad_internal,
            "variable": self.variable,
        }


# ── Operation Spec (Layer A: pure data) ──

@dataclass(frozen=True)
class OperationSpec:
    """Canonical specification of a single digit operation. Pure data, no callables."""
    name: str
    display_name: str
    description: str
    semantic_class: SemanticClass
    ds_class: DSClass
    digit_length: DigitLengthSpec
    leading_zero_policy: LeadingZeroPolicy
    preserves_mod_b_minus_1: bool = False
    preserves_mod_b_plus_1: bool = False
    arity: int = 1                      # Unary for now; future-proofed
    base_dependent: bool = False
    parameters: Tuple[str, ...] = ()
    default_params: Dict[str, Any] = field(default_factory=dict)
    invariants: Tuple[str, ...] = ()    # Known algebraic invariants
    monotonicity: str = "none"          # "decreasing" | "non_increasing" | "none"
    version: str = "1.0"

    def canonical_dict(self) -> dict:
        """Full canonical dict for registry hashing."""
        return {
            "name": self.name,
            "semantic_class": self.semantic_class.value,
            "ds_class": self.ds_class.value,
            "digit_length": self.digit_length.canonical_dict(),
            "leading_zero_policy": self.leading_zero_policy.value,
            "preserves_mod_b_minus_1": self.preserves_mod_b_minus_1,
            "preserves_mod_b_plus_1": self.preserves_mod_b_plus_1,
            "arity": self.arity,
            "base_dependent": self.base_dependent,
            "parameters": list(self.parameters),
            "default_params": dict(sorted(self.default_params.items())) if self.default_params else {},
            "invariants": list(self.invariants),
            "monotonicity": self.monotonicity,
            "version": self.version,
        }

    def step_dict(self, params: Optional[Dict[str, Any]] = None) -> dict:
        """Minimal dict for pipeline step hashing."""
        d: dict = {"op": self.name}
        if params:
            d["params"] = dict(sorted(params.items()))
        elif self.default_params:
            d["params"] = dict(sorted(self.default_params.items()))
        return d


# ── Pipeline Spec (Layer A) ──

@dataclass(frozen=True)
class PipelineStep:
    """A single step in a pipeline: operation name + parameters."""
    op_name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict:
        d: dict = {"op": self.op_name}
        if self.params:
            d["params"] = dict(sorted(self.params.items()))
        return d

    def display(self) -> str:
        if self.params:
            p = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
            return f"{self.op_name}({p})"
        return self.op_name


@dataclass(frozen=True)
class Pipeline:
    """Canonical pipeline: ordered tuple of operations with SHA-256 identity.

    Hashing: DSL string -> parse -> canonical JSON (sorted keys) -> SHA-256.
    Never hash the string directly.
    """
    steps: Tuple[PipelineStep, ...]
    version: str = "2.0"

    @staticmethod
    def parse(dsl_string: str, registry: Optional[OperationRegistry] = None) -> Pipeline:
        """Parse DSL string like 'kaprekar_step |> digit_pow4 |> digit_sum'.

        Separators: |> , -> , >>
        Parameters: op_name(key=value, key2=value2)
        Whitespace is normalized (never affects hash).
        """
        dsl_string = dsl_string.strip()
        parts = re.split(r'\s*(?:\|>|->|>>)\s*', dsl_string)

        steps = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = re.match(r'^(\w+)(?:\(([^)]*)\))?$', part)
            if not m:
                raise ValueError(f"Invalid pipeline step: {part!r}")
            op_name = m.group(1)
            params: Dict[str, Any] = {}
            if m.group(2):
                for kv in m.group(2).split(','):
                    kv = kv.strip()
                    if '=' in kv:
                        k, v_str = kv.split('=', 1)
                        k, v_str = k.strip(), v_str.strip()
                        v: Any = v_str
                        try:
                            v = int(v_str)
                        except ValueError:
                            try:
                                v = float(v_str)
                            except ValueError:
                                pass
                        params[k] = v

            if registry is not None:
                registry.get_spec(op_name)

            steps.append(PipelineStep(op_name=op_name, params=params))

        return Pipeline(steps=tuple(steps))

    @staticmethod
    def from_list(op_names: List[str], registry: Optional[OperationRegistry] = None) -> Pipeline:
        steps = []
        for name in op_names:
            if registry is not None:
                registry.get_spec(name)
            steps.append(PipelineStep(op_name=name))
        return Pipeline(steps=tuple(steps))

    @staticmethod
    def from_json(data: dict) -> Pipeline:
        steps = []
        for step_data in data.get("pipeline", data.get("steps", [])):
            steps.append(PipelineStep(op_name=step_data["op"], params=step_data.get("params", {})))
        return Pipeline(steps=tuple(steps), version=data.get("version", "2.0"))

    def canonical_dict(self) -> dict:
        return {
            "pipeline": [s.canonical_dict() for s in self.steps],
            "version": self.version,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(',', ':'))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.sha256[:16]

    def display(self) -> str:
        return " |> ".join(s.display() for s in self.steps)

    def to_json(self) -> str:
        return json.dumps(self.canonical_dict(), indent=2, sort_keys=True)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"Pipeline({self.display()!r}, hash={self.short_hash})"


# ── Domain Policy (Layer A) ──

@dataclass(frozen=True)
class DomainPolicy:
    """Canonical specification of the input domain for a pipeline run."""
    base: int = 10
    digit_length: Optional[int] = None
    digit_length_min: Optional[int] = None
    digit_length_max: Optional[int] = None
    leading_zero_policy: LeadingZeroPolicy = LeadingZeroPolicy.DROPS
    exclude_repdigits: bool = False
    exclude_zero: bool = True
    include_cycles: bool = True         # Count cycles as attractors in analysis
    explicit_range: Optional[Tuple[int, int]] = None
    engine_semantic_version: str = "2.0"
    version: str = "1.0"
    preset_name: Optional[str] = None   # Explicit label: "paper_a_kaprekar", etc.

    def range(self) -> Tuple[int, int]:
        if self.explicit_range is not None:
            return self.explicit_range
        if self.digit_length is not None:
            k = self.digit_length
            lo = self.base ** (k - 1) if k > 1 else (1 if self.exclude_zero else 0)
            hi = self.base ** k - 1
            return (lo, hi)
        if self.digit_length_min is not None and self.digit_length_max is not None:
            lo = self.base ** (self.digit_length_min - 1)
            hi = self.base ** self.digit_length_max - 1
            return (lo, hi)
        raise ValueError("DomainPolicy must specify digit_length, digit_length_min/max, or explicit_range")

    def count(self) -> int:
        lo, hi = self.range()
        c = hi - lo + 1
        if self.exclude_repdigits and self.digit_length is not None:
            c -= (self.base - 1)
        return c

    @staticmethod
    def is_repdigit(n: int) -> bool:
        s = str(n)
        return len(set(s)) == 1

    def iterate(self):
        lo, hi = self.range()
        for n in range(lo, hi + 1):
            if self.exclude_repdigits and self.is_repdigit(n):
                continue
            yield n

    def canonical_dict(self) -> dict:
        d: dict = {
            "base": self.base,
            "leading_zero_policy": self.leading_zero_policy.value,
            "exclude_repdigits": self.exclude_repdigits,
            "exclude_zero": self.exclude_zero,
            "include_cycles": self.include_cycles,
            "engine_semantic_version": self.engine_semantic_version,
            "version": self.version,
        }
        if self.digit_length is not None:
            d["digit_length"] = self.digit_length
        if self.digit_length_min is not None:
            d["digit_length_min"] = self.digit_length_min
        if self.digit_length_max is not None:
            d["digit_length_max"] = self.digit_length_max
        if self.explicit_range is not None:
            d["explicit_range"] = list(self.explicit_range)
        if self.preset_name is not None:
            d["preset_name"] = self.preset_name
        return d

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(',', ':'))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.sha256[:16]

    def __repr__(self) -> str:
        lo, hi = self.range()
        return f"Domain(base={self.base}, range=[{lo},{hi}], hash={self.short_hash})"

    # ── Named presets (punt 1: Paper A vs Engine Metrics definitions) ──

    @classmethod
    def paper_a_kaprekar(cls, k: int, base: int = 10) -> DomainPolicy:
        """Paper A definition: k-digit non-repdigit domain, exclude_zero=True.
        For d=4: all 4-digit non-repdigit → 100% convergence to 6174."""
        return cls(base=base, digit_length=k, exclude_repdigits=True,
                   exclude_zero=True, include_cycles=False,
                   engine_semantic_version="2.0",
                   preset_name="paper_a_kaprekar")

    @classmethod
    def engine_metrics_kaprekar(cls, k: int, base: int = 10) -> DomainPolicy:
        """Engine metrics definition: includes 0 as attractor, cycles counted.
        convergence_rate = dominant basin fraction (not 'convergence to FP')."""
        return cls(base=base, digit_length=k, exclude_repdigits=True,
                   exclude_zero=True, include_cycles=True,
                   engine_semantic_version="2.0",
                   preset_name="engine_metrics_kaprekar")

    @classmethod
    def paper_a_1089(cls, k: int, base: int = 10) -> DomainPolicy:
        """Paper A definition for 1089-trick: k-digit domain, non-palindromes."""
        return cls(base=base, digit_length=k, exclude_repdigits=False,
                   exclude_zero=True, include_cycles=False,
                   engine_semantic_version="2.0",
                   preset_name="paper_a_1089")

    @classmethod
    def engine_metrics_1089(cls, k: int, base: int = 10) -> DomainPolicy:
        """Engine metrics definition for 1089: full domain, all attractors."""
        return cls(base=base, digit_length=k, exclude_repdigits=False,
                   exclude_zero=True, include_cycles=True,
                   engine_semantic_version="2.0",
                   preset_name="engine_metrics_1089")


# =============================================================================
# LAYER A — RESULT SPEC (pure data)
# =============================================================================

FLOAT_PRECISION = 12  # Digits for canonical float formatting

def canonical_float(x: float) -> str:
    """Fixed-precision float formatting for deterministic hashing."""
    return format(x, f".{FLOAT_PRECISION}f")


@dataclass
class WitnessTrace:
    """A single orbit trace from start to attractor."""
    start: int
    attractor: int
    steps: int
    orbit: Optional[List[int]] = None

    def canonical_dict(self) -> dict:
        d: dict = {"start": self.start, "attractor": self.attractor, "steps": self.steps}
        if self.orbit is not None:
            d["orbit"] = self.orbit
        return d


@dataclass
class RunResult:
    """Deterministic, hashable result of a pipeline run over a domain."""
    pipeline_hash: str
    pipeline_display: str
    domain_hash: str
    domain_display: str
    op_registry_hash: str           # Hash of the operation registry used
    num_startpoints: int
    fixed_points: List[int]
    cycles: List[List[int]]
    cycle_lengths: List[int]
    num_attractors: int
    basin_fractions: Dict[str, float]
    avg_steps: float
    max_steps: int
    median_steps: float
    convergence_rate: float
    basin_entropy: float
    witnesses: List[WitnessTrace] = field(default_factory=list)
    engine_version: str = "16.0"
    timestamp: str = ""

    def canonical_dict(self) -> dict:
        """Deterministic dict for hashing.
        Floats use fixed precision. Excludes witnesses and timestamp."""
        bf_canonical = {}
        for k, v in sorted(self.basin_fractions.items()):
            bf_canonical[k] = canonical_float(v)
        return {
            "pipeline_hash": self.pipeline_hash,
            "domain_hash": self.domain_hash,
            "op_registry_hash": self.op_registry_hash,
            "num_startpoints": self.num_startpoints,
            "fixed_points": sorted(self.fixed_points),
            "cycles": sorted([sorted(c) for c in self.cycles]),
            "cycle_lengths": sorted(self.cycle_lengths),
            "num_attractors": self.num_attractors,
            "basin_fractions": bf_canonical,
            "avg_steps": canonical_float(self.avg_steps),
            "max_steps": self.max_steps,
            "median_steps": canonical_float(self.median_steps),
            "convergence_rate": canonical_float(self.convergence_rate),
            "basin_entropy": canonical_float(self.basin_entropy),
            "engine_version": self.engine_version,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(',', ':'))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.sha256[:16]

    def to_json(self, include_witnesses: bool = False) -> str:
        d = self.canonical_dict()
        d["timestamp"] = self.timestamp
        d["pipeline_display"] = self.pipeline_display
        d["domain_display"] = self.domain_display
        d["result_hash"] = self.sha256
        if include_witnesses:
            d["witnesses"] = [w.canonical_dict() for w in self.witnesses]
        return json.dumps(d, indent=2, sort_keys=True)

    @staticmethod
    def compute_basin_entropy(basin_fractions: Dict[str, float]) -> float:
        h = 0.0
        for p in basin_fractions.values():
            if p > 0:
                h -= p * math.log2(p)
        return h


# =============================================================================
# LAYER B — EXECUTION LAYER (implementations, separate from specs)
# =============================================================================

class OperationExecutor:
    """Maps operation names to concrete implementations. Layer B only."""

    @staticmethod
    def reverse(n: int) -> int:
        return int(str(abs(n))[::-1]) if n != 0 else 0

    @staticmethod
    def digit_sum(n: int) -> int:
        return sum(int(d) for d in str(abs(n)))

    @staticmethod
    def digit_product(n: int) -> int:
        r = 1
        for d in str(abs(n)):
            if int(d) > 0:
                r *= int(d)
        return r

    @staticmethod
    def make_digit_pow(p: int) -> Callable:
        def _dp(n: int) -> int:
            return sum(int(d) ** p for d in str(abs(n)))
        return _dp

    @staticmethod
    def sort_asc(n: int) -> int:
        s = ''.join(sorted(str(abs(n)))).lstrip('0')
        return int(s) if s else 0

    @staticmethod
    def sort_desc(n: int) -> int:
        return int(''.join(sorted(str(abs(n)), reverse=True)))

    @staticmethod
    def kaprekar_step(n: int) -> int:
        desc = int(''.join(sorted(str(abs(n)), reverse=True)))
        s = ''.join(sorted(str(abs(n)))).lstrip('0')
        asc = int(s) if s else 0
        return desc - asc

    @staticmethod
    def truc_1089(n: int) -> int:
        if n <= 0:
            return 0
        rev = int(str(n)[::-1])
        diff = abs(n - rev)
        if diff == 0:
            return 0
        return diff + int(str(diff)[::-1])

    @staticmethod
    def swap_ends(n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        return int((s[-1] + s[1:-1] + s[0]).lstrip('0') or '0')

    @staticmethod
    def complement_9(n: int) -> int:
        return int(''.join(str(9 - int(d)) for d in str(abs(n))).lstrip('0') or '0')

    @staticmethod
    def add_reverse(n: int) -> int:
        return abs(n) + int(str(abs(n))[::-1])

    @staticmethod
    def sub_reverse(n: int) -> int:
        return abs(abs(n) - int(str(abs(n))[::-1]))

    @staticmethod
    def digit_factorial_sum(n: int) -> int:
        f = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        return sum(f[int(d)] for d in str(abs(n)))

    @staticmethod
    def collatz_step(n: int) -> int:
        if n <= 0:
            return 0
        return n // 2 if n % 2 == 0 else 3 * n + 1

    @staticmethod
    def rotate_left(n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        return int((s[1:] + s[0]).lstrip('0') or '0')

    @staticmethod
    def rotate_right(n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        return int((s[-1] + s[:-1]).lstrip('0') or '0')

    @staticmethod
    def digit_gcd(n: int) -> int:
        digits = [int(d) for d in str(abs(n)) if d != '0']
        if not digits:
            return 0
        result = digits[0]
        for d in digits[1:]:
            result = math.gcd(result, d)
        return result

    @staticmethod
    def digit_xor(n: int) -> int:
        result = 0
        for d in str(abs(n)):
            result ^= int(d)
        return result

    @staticmethod
    def narcissistic_step(n: int) -> int:
        s = str(abs(n))
        k = len(s)
        return sum(int(d) ** k for d in s)


# =============================================================================
# OPERATION REGISTRY (bridges Layer A specs + Layer B implementations)
# =============================================================================

# Shorthand constructors
_DL = DigitLengthSpec
_P = LeadingZeroPolicy
_S = SemanticClass
_DS = DSClass

# All 22 operation specifications (Layer A: pure data)
_ALL_SPECS: List[OperationSpec] = [
    # ── ds-preserving (P): permutations ──
    OperationSpec("reverse", "rev", "Reverse digit string; leading zeros drop.",
                  _S.PERMUTATION, _DS.PRESERVING,
                  _DL(may_reduce=True), _P.DROPS,
                  preserves_mod_b_minus_1=True, preserves_mod_b_plus_1=False,
                  invariants=("digit_multiset", "digit_sum", "mod_9")),
    OperationSpec("sort_asc", "sort↑", "Sort digits ascending; leading zeros drop.",
                  _S.PERMUTATION, _DS.PRESERVING,
                  _DL(may_reduce=True), _P.DROPS,
                  preserves_mod_b_minus_1=True,
                  invariants=("digit_multiset", "digit_sum", "mod_9")),
    OperationSpec("sort_desc", "sort↓", "Sort digits descending; preserves digit count.",
                  _S.PERMUTATION, _DS.PRESERVING,
                  _DL(fixed=True), _P.PRESERVES,
                  preserves_mod_b_minus_1=True,
                  invariants=("digit_multiset", "digit_sum", "mod_9"),
                  monotonicity="non_increasing"),
    OperationSpec("rotate_left", "rot_l", "Rotate digits left by 1; leading zeros drop.",
                  _S.PERMUTATION, _DS.PRESERVING,
                  _DL(may_reduce=True), _P.DROPS,
                  preserves_mod_b_minus_1=True,
                  invariants=("digit_multiset", "digit_sum")),
    OperationSpec("rotate_right", "rot_r", "Rotate digits right by 1; leading zeros drop.",
                  _S.PERMUTATION, _DS.PRESERVING,
                  _DL(may_reduce=True), _P.DROPS,
                  preserves_mod_b_minus_1=True,
                  invariants=("digit_multiset", "digit_sum")),
    OperationSpec("swap_ends", "swap", "Swap first and last digits; leading zeros drop.",
                  _S.PERMUTATION, _DS.PRESERVING,
                  _DL(may_reduce=True), _P.DROPS,
                  preserves_mod_b_minus_1=True,
                  invariants=("digit_multiset", "digit_sum")),
    # ── ds-contractive (C): aggregates ──
    OperationSpec("digit_sum", "ds", "Sum of digits.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  preserves_mod_b_minus_1=True,
                  invariants=("mod_9",), monotonicity="decreasing"),
    OperationSpec("digit_product", "dp", "Product of nonzero digits.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  monotonicity="decreasing"),
    OperationSpec("digit_gcd", "dgcd", "GCD of all nonzero digits.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  monotonicity="decreasing"),
    OperationSpec("digit_xor", "dxor", "XOR of all digits.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA),
    # ── ds-contractive (C): digit-power maps ──
    OperationSpec("digit_pow2", "dp2", "Sum of digits^2.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  parameters=("p",), default_params={"p": 2},
                  monotonicity="decreasing"),
    OperationSpec("digit_pow3", "dp3", "Sum of digits^3.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  parameters=("p",), default_params={"p": 3},
                  monotonicity="decreasing"),
    OperationSpec("digit_pow4", "dp4", "Sum of digits^4.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  parameters=("p",), default_params={"p": 4},
                  monotonicity="decreasing"),
    OperationSpec("digit_pow5", "dp5", "Sum of digits^5.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  parameters=("p",), default_params={"p": 5},
                  monotonicity="decreasing"),
    OperationSpec("digit_factorial_sum", "dfac", "Sum of factorials of each digit.",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  monotonicity="decreasing"),
    OperationSpec("narcissistic_step", "narc", "Sum of d_i^k where k = #digits(n).",
                  _S.AGGREGATE, _DS.CONTRACTIVE,
                  _DL(reduces_to_small=True), _P.NA,
                  monotonicity="decreasing"),
    # ── ds-expansive (X) ──
    OperationSpec("complement_9", "comp", "Replace each digit d by 9-d; leading zeros drop.",
                  _S.COMPLEMENT, _DS.EXPANSIVE,
                  _DL(may_reduce=True), _P.DROPS,
                  base_dependent=True, default_params={"base": 10},
                  invariants=("complement_involution",)),
    OperationSpec("kaprekar_step", "kap", "sort_desc(n) - sort_asc(n); zero-pads to k digits.",
                  _S.SUBTRACTIVE, _DS.EXPANSIVE,
                  _DL(may_reduce=True, pad_internal=True), _P.PADS,
                  preserves_mod_b_minus_1=True, base_dependent=True,
                  default_params={"base": 10},
                  invariants=("mod_9", "digit_sum_zero_diff")),
    OperationSpec("truc_1089", "1089", "|n - rev(n)| + rev(|n - rev(n)|); 0 if palindrome.",
                  _S.MIXED, _DS.EXPANSIVE,
                  _DL(may_increase=True, may_reduce=True), _P.DROPS,
                  preserves_mod_b_minus_1=True,
                  invariants=("mod_9",)),
    OperationSpec("add_reverse", "addr", "n + rev(n).",
                  _S.MIXED, _DS.EXPANSIVE,
                  _DL(may_increase=True), _P.NA,
                  preserves_mod_b_minus_1=True,
                  invariants=("mod_9",)),
    OperationSpec("sub_reverse", "subr", "|n - rev(n)|.",
                  _S.MIXED, _DS.EXPANSIVE,
                  _DL(may_reduce=True), _P.DROPS,
                  preserves_mod_b_minus_1=True,
                  invariants=("mod_9",)),
    OperationSpec("collatz_step", "col", "n/2 if even, 3n+1 if odd.",
                  _S.ARITHMETIC, _DS.EXPANSIVE,
                  _DL(variable=True), _P.NA),
]

# Layer B: map names to implementations
_IMPL_MAP: Dict[str, Callable] = {
    "reverse": OperationExecutor.reverse,
    "sort_asc": OperationExecutor.sort_asc,
    "sort_desc": OperationExecutor.sort_desc,
    "rotate_left": OperationExecutor.rotate_left,
    "rotate_right": OperationExecutor.rotate_right,
    "swap_ends": OperationExecutor.swap_ends,
    "digit_sum": OperationExecutor.digit_sum,
    "digit_product": OperationExecutor.digit_product,
    "digit_gcd": OperationExecutor.digit_gcd,
    "digit_xor": OperationExecutor.digit_xor,
    "digit_pow2": OperationExecutor.make_digit_pow(2),
    "digit_pow3": OperationExecutor.make_digit_pow(3),
    "digit_pow4": OperationExecutor.make_digit_pow(4),
    "digit_pow5": OperationExecutor.make_digit_pow(5),
    "digit_factorial_sum": OperationExecutor.digit_factorial_sum,
    "narcissistic_step": OperationExecutor.narcissistic_step,
    "complement_9": OperationExecutor.complement_9,
    "kaprekar_step": OperationExecutor.kaprekar_step,
    "truc_1089": OperationExecutor.truc_1089,
    "add_reverse": OperationExecutor.add_reverse,
    "sub_reverse": OperationExecutor.sub_reverse,
    "collatz_step": OperationExecutor.collatz_step,
}


class OperationRegistry:
    """Bridges Layer A (specs) and Layer B (implementations).

    The registry itself is hashable: any change to specs changes the hash.
    """

    VERSION = "2.0"

    def __init__(self):
        self._specs: Dict[str, OperationSpec] = {s.name: s for s in _ALL_SPECS}
        self._impls: Dict[str, Callable] = dict(_IMPL_MAP)

    def get_spec(self, name: str) -> OperationSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown operation: {name!r}. Known: {sorted(self._specs.keys())}")
        return self._specs[name]

    def get_impl(self, name: str) -> Callable:
        if name not in self._impls:
            raise KeyError(f"No implementation for: {name!r}")
        return self._impls[name]

    def execute(self, name: str, n: int) -> int:
        return self.get_impl(name)(n)

    def all_names(self) -> List[str]:
        return sorted(self._specs.keys())

    def all_specs(self) -> List[OperationSpec]:
        return [self._specs[n] for n in self.all_names()]

    def canonical_dict(self) -> dict:
        """Full registry as canonical dict (for hashing)."""
        return {
            "version": self.VERSION,
            "operations": {name: self._specs[name].canonical_dict()
                           for name in self.all_names()},
        }

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(',', ':'))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.sha256[:16]

    def summary_table(self) -> str:
        lines = [
            f"{'Name':<22} {'Disp':<6} {'Sem':<14} {'DS':<3} "
            f"{'fix':<4} {'↓':<3} {'↑':<3} {'sm':<3} {'pad':<4} "
            f"{'mod9':<5} {'LZ':<10} {'Inv'}"
        ]
        lines.append("-" * 100)
        for spec in self.all_specs():
            dl = spec.digit_length
            lines.append(
                f"{spec.name:<22} {spec.display_name:<6} "
                f"{spec.semantic_class.value:<14} {spec.ds_class.value:<3} "
                f"{'Y' if dl.fixed else '.':<4} "
                f"{'Y' if dl.may_reduce else '.':<3} "
                f"{'Y' if dl.may_increase else '.':<3} "
                f"{'Y' if dl.reduces_to_small else '.':<3} "
                f"{'Y' if dl.pad_internal else '.':<4} "
                f"{'Y' if spec.preserves_mod_b_minus_1 else '.':<5} "
                f"{spec.leading_zero_policy.value:<10} "
                f"{','.join(spec.invariants)}"
            )
        return "\n".join(lines)

    def execute_pipeline(self, pipeline: Pipeline, n: int) -> int:
        for step in pipeline.steps:
            n = self.get_impl(step.op_name)(n)
        return n


# =============================================================================
# PIPELINE RUNNER (Layer B: execution + result construction)
# =============================================================================

class PipelineRunner:
    """Executes pipelines over domains and produces deterministic RunResults."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()

    def run_exhaustive(
        self,
        pipeline: Pipeline,
        domain: DomainPolicy,
        max_iter: int = 200,
        store_witnesses: int = 10,
    ) -> RunResult:
        """Run pipeline exhaustively over domain. Fully deterministic."""
        from datetime import datetime

        reg = self.registry
        attractor_counts: Dict[int, int] = {}
        step_counts: List[int] = []
        witness_traces: List[WitnessTrace] = []
        cycles_found: Dict[int, List[int]] = {}
        fixed_points: Set[int] = set()

        total = 0
        for n in domain.iterate():
            total += 1
            val = n
            seen = {val: 0}
            orbit = [val]
            converged = False
            for t in range(1, max_iter + 1):
                val = reg.execute_pipeline(pipeline, val)
                orbit.append(val)
                if val in seen:
                    cycle_start = seen[val]
                    cycle = orbit[cycle_start:t]
                    if len(cycle) == 1:
                        fixed_points.add(val)
                    else:
                        cycle_key = min(cycle)
                        if cycle_key not in cycles_found:
                            cycles_found[cycle_key] = sorted(set(cycle))
                    attractor_counts[val] = attractor_counts.get(val, 0) + 1
                    step_counts.append(t)
                    converged = True
                    break
                if val == 0:
                    attractor_counts[0] = attractor_counts.get(0, 0) + 1
                    step_counts.append(t)
                    converged = True
                    fixed_points.add(0)
                    break
                seen[val] = t

            if not converged:
                attractor_counts[-1] = attractor_counts.get(-1, 0) + 1
                step_counts.append(max_iter)

            if len(witness_traces) < store_witnesses:
                witness_traces.append(WitnessTrace(
                    start=n, attractor=val, steps=step_counts[-1],
                    orbit=orbit[:min(20, len(orbit))],
                ))

        if not step_counts:
            step_counts = [0]

        sorted_steps = sorted(step_counts)
        median_steps = sorted_steps[len(sorted_steps) // 2]

        basin_fracs: Dict[str, float] = {}
        for att, cnt in attractor_counts.items():
            basin_fracs[str(att)] = cnt / total if total > 0 else 0.0

        dominant_count = max(attractor_counts.values()) if attractor_counts else 0
        convergence_rate = dominant_count / total if total > 0 else 0.0
        basin_entropy = RunResult.compute_basin_entropy(basin_fracs)

        return RunResult(
            pipeline_hash=pipeline.sha256,
            pipeline_display=pipeline.display(),
            domain_hash=domain.sha256,
            domain_display=repr(domain),
            op_registry_hash=reg.sha256,
            num_startpoints=total,
            fixed_points=sorted(fixed_points),
            cycles=[sorted(c) for c in cycles_found.values()],
            cycle_lengths=sorted(len(c) for c in cycles_found.values()),
            num_attractors=len(attractor_counts),
            basin_fractions=basin_fracs,
            avg_steps=sum(step_counts) / len(step_counts),
            max_steps=max(step_counts),
            median_steps=float(median_steps),
            convergence_rate=convergence_rate,
            basin_entropy=basin_entropy,
            witnesses=witness_traces,
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# CONVENIENCE
# =============================================================================

def quick_run(
    dsl: str,
    digit_length: int = 4,
    base: int = 10,
    exclude_repdigits: bool = True,
    max_iter: int = 200,
) -> RunResult:
    """One-liner: parse DSL, build domain, run exhaustively, return result."""
    reg = OperationRegistry()
    pipe = Pipeline.parse(dsl, registry=reg)
    domain = DomainPolicy(base=base, digit_length=digit_length, exclude_repdigits=exclude_repdigits)
    runner = PipelineRunner(reg)
    return runner.run_exhaustive(pipe, domain, max_iter=max_iter)


# =============================================================================
# MAIN: Self-test & demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENGINE vNext — M0 v2.0: Canonical Semantics & Reproducibility")
    print("=" * 70)

    reg = OperationRegistry()
    runner = PipelineRunner(reg)

    # ── 1. Operation Registry ──
    print(f"\n── 1. Operation Registry ({len(reg.all_names())} ops, hash={reg.short_hash}) ──\n")
    print(reg.summary_table())

    # ── 2. Pipeline DSL ──
    print("\n── 2. Pipeline DSL (whitespace-invariant hashing) ──\n")
    for dsl in [
        "kaprekar_step |> digit_pow4 |> digit_sum",
        "truc_1089 |> digit_pow4",
        "complement_9 |> reverse",
        "sort_desc",
    ]:
        p = Pipeline.parse(dsl, registry=reg)
        print(f"  {dsl:<50} hash={p.short_hash}")

    # Whitespace invariance
    p1 = Pipeline.parse("kaprekar_step|>digit_sum", registry=reg)
    p2 = Pipeline.parse("kaprekar_step  |>  digit_sum", registry=reg)
    print(f"\n  Whitespace invariance: {p1.sha256 == p2.sha256}")

    # Order sensitivity
    p3 = Pipeline.parse("digit_sum |> kaprekar_step", registry=reg)
    print(f"  Order sensitivity:     {p1.sha256 != p3.sha256}")

    # ── 3. Domain Policy ──
    print("\n── 3. Domain Policy ──\n")
    for d in [
        DomainPolicy(base=10, digit_length=4, exclude_repdigits=True),
        DomainPolicy(base=10, digit_length=6, exclude_repdigits=True),
    ]:
        lo, hi = d.range()
        print(f"  {d}  count={d.count()}")

    # ── 4. Kaprekar 4-digit (golden test candidate) ──
    print("\n── 4. Kaprekar 4-digit ──\n")
    pipe = Pipeline.parse("kaprekar_step", registry=reg)
    domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
    result = runner.run_exhaustive(pipe, domain)
    print(f"  FPs:        {result.fixed_points}")
    print(f"  Cycles:     {result.cycle_lengths}")
    print(f"  Conv rate:  {result.convergence_rate:.6f}")
    print(f"  Avg steps:  {result.avg_steps:.2f}")
    print(f"  Basin H:    {result.basin_entropy:.6f} bits")
    print(f"  Result SHA: {result.short_hash}")

    # ── 5. truc_1089 5-digit ──
    print("\n── 5. truc_1089 5-digit ──\n")
    pipe = Pipeline.parse("truc_1089", registry=reg)
    domain = DomainPolicy(base=10, digit_length=5)
    result = runner.run_exhaustive(pipe, domain)
    print(f"  FPs:        {result.fixed_points}")
    print(f"  Conv rate:  {result.convergence_rate:.6f}")
    print(f"  Result SHA: {result.short_hash}")

    # ── 6. Registry hash stability ──
    print("\n── 6. Registry hash ──\n")
    reg2 = OperationRegistry()
    print(f"  Registry 1: {reg.sha256[:32]}...")
    print(f"  Registry 2: {reg2.sha256[:32]}...")
    print(f"  Match:      {reg.sha256 == reg2.sha256}")

    print("\n" + "=" * 70)
    print("M0 v2.0 SELF-TEST COMPLETE")
    print("=" * 70)
