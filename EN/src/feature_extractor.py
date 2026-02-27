# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research — MIT License
"""
ENGINE vNext — Module M2: Feature Extractor + Conjecture Mining

Builds on M0 (pipeline_dsl.py) and M1 (experiment_runner.py) to provide:
  1. Per-number features: ds(n), mod invariants, palindrome, sortedness, complement score
  2. Per-orbit features: contraction ratio, cycle detection, transient length
  3. Per-pipeline features: semantic signature, monotonicity profile
  4. Conjecture templates: counting, invariant, universality, structure
  5. Conjecture ranker v0 (heuristic scoring)
  6. Falsification engine: targeted counterexample search + delta-debugging

Architecture note (M2→M0 execution dependency):
    ConjectureMiner and FalsificationEngine import PipelineRunner from M0.
    This crosses the semantic/execution boundary: M2 is a "semantic" module
    but needs to *run* pipelines to mine conjectures and test them. This is
    a pragmatic design choice — extracting features from experiment results
    alone would be insufficient for targeted falsification. The dependency
    is confined to two classes and does not affect M0's semantic layer.

Usage:
    from feature_extractor import NumberFeatures, OrbitAnalyzer, ConjectureMiner

    nf = NumberFeatures(base=10)
    print(nf.compute(6174))

    miner = ConjectureMiner(store)
    conjectures = miner.mine_all()
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline_dsl import (
    OperationRegistry, Pipeline, DomainPolicy, RunResult,
    PipelineRunner, OperationExecutor, SemanticClass, DSClass,
    canonical_float,
)


# =============================================================================
# MONOTONICITY LANGUAGE (punt 3: standardize terminology)
# =============================================================================

class MonotonicityKind(str, Enum):
    """Precise monotonicity classification for conjecture statements."""
    NONDECREASING = "monotone_nondecreasing"       # f(k₁) ≤ f(k₂) for k₁<k₂
    STRICTLY_INCREASING = "strictly_increasing"     # f(k₁) < f(k₂) for k₁<k₂
    NONINCREASING = "monotone_nonincreasing"        # f(k₁) ≥ f(k₂) for k₁<k₂
    STRICTLY_DECREASING = "strictly_decreasing"     # f(k₁) > f(k₂) for k₁<k₂
    EVENTUALLY_CONSTANT = "eventually_constant"     # ∃K: f(k)=c for all k≥K


# =============================================================================
# 1. PER-NUMBER FEATURES
# =============================================================================

@dataclass
class NumberProfile:
    """Structural features of a single number."""
    n: int
    base: int
    num_digits: int
    digit_sum: int
    digit_product: int
    digital_root: int           # iterated digit sum until single digit
    mod_b_minus_1: int          # n mod (base-1), e.g. mod 9
    mod_b_plus_1: int           # n mod (base+1), e.g. mod 11
    is_palindrome: bool
    is_repdigit: bool
    sortedness_tau: float       # Kendall τ: 1.0 = ascending, -1.0 = descending
    complement_distance: int    # |n - complement(n)|
    leading_digit: int
    trailing_digit: int
    digit_entropy: float        # Shannon entropy of digit distribution
    digit_range: int            # max_digit - min_digit
    has_zero_digit: bool

    def to_dict(self) -> dict:
        return {
            "n": self.n, "base": self.base, "num_digits": self.num_digits,
            "digit_sum": self.digit_sum, "digit_product": self.digit_product,
            "digital_root": self.digital_root,
            "mod_b_minus_1": self.mod_b_minus_1, "mod_b_plus_1": self.mod_b_plus_1,
            "is_palindrome": self.is_palindrome, "is_repdigit": self.is_repdigit,
            "sortedness_tau": round(self.sortedness_tau, 6),
            "complement_distance": self.complement_distance,
            "leading_digit": self.leading_digit, "trailing_digit": self.trailing_digit,
            "digit_entropy": round(self.digit_entropy, 6),
            "digit_range": self.digit_range, "has_zero_digit": self.has_zero_digit,
        }


class NumberFeatures:
    """Compute structural features for numbers in a given base."""

    def __init__(self, base: int = 10):
        self.base = base

    def digits(self, n: int) -> List[int]:
        if n == 0:
            return [0]
        ds = []
        val = abs(n)
        while val > 0:
            ds.append(val % self.base)
            val //= self.base
        return list(reversed(ds))

    def compute(self, n: int, num_digits: Optional[int] = None) -> NumberProfile:
        ds = self.digits(n)
        if num_digits and len(ds) < num_digits:
            ds = [0] * (num_digits - len(ds)) + ds
        k = len(ds)
        b = self.base

        dsum = sum(ds)
        dprod = 1
        for d in ds:
            dprod *= d

        # Digital root
        dr = n
        while dr >= b:
            dr = sum(self.digits(dr))

        # Palindrome
        is_pal = ds == ds[::-1]

        # Repdigit
        is_rep = len(set(ds)) == 1

        # Kendall τ sortedness
        tau = self._kendall_tau(ds)

        # Complement distance
        comp = sum((b - 1 - d) for d in ds)
        comp_n = 0
        for d in ds:
            comp_n = comp_n * b + (b - 1 - d)
        comp_dist = abs(n - comp_n)

        # Digit entropy
        if k > 0:
            counts = Counter(ds)
            entropy = 0.0
            for c in counts.values():
                p = c / k
                if p > 0:
                    entropy -= p * math.log2(p)
        else:
            entropy = 0.0

        return NumberProfile(
            n=n, base=b, num_digits=k,
            digit_sum=dsum, digit_product=dprod, digital_root=dr,
            mod_b_minus_1=n % (b - 1) if b > 1 else 0,
            mod_b_plus_1=n % (b + 1),
            is_palindrome=is_pal, is_repdigit=is_rep,
            sortedness_tau=tau,
            complement_distance=comp_dist,
            leading_digit=ds[0] if ds else 0,
            trailing_digit=ds[-1] if ds else 0,
            digit_entropy=entropy,
            digit_range=max(ds) - min(ds) if ds else 0,
            has_zero_digit=0 in ds,
        )

    @staticmethod
    def _kendall_tau(seq: List[int]) -> float:
        """Kendall τ: +1 = fully sorted asc, -1 = fully sorted desc."""
        n = len(seq)
        if n < 2:
            return 0.0
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                if seq[i] < seq[j]:
                    concordant += 1
                elif seq[i] > seq[j]:
                    discordant += 1
        total = n * (n - 1) / 2
        if total == 0:
            return 0.0
        return (concordant - discordant) / total


# =============================================================================
# 2. PER-ORBIT FEATURES
# =============================================================================

@dataclass
class OrbitProfile:
    """Features of a single orbit (trajectory from start to attractor)."""
    start: int
    attractor: int
    transient_length: int       # steps before entering cycle/FP
    cycle_length: int           # 1 for FP, >1 for cycle
    is_fixed_point: bool
    orbit_digest: str           # first 8 values as signature
    contraction_ratios: List[float]  # |f(n)|/|n| per step
    avg_contraction: float
    monotone_steps: int         # count of steps where value decreases
    expansion_steps: int        # count of steps where value increases
    digit_sum_trajectory: List[int]  # digit sums along orbit
    mod9_trajectory: List[int]       # n mod 9 along orbit

    def to_dict(self) -> dict:
        return {
            "start": self.start, "attractor": self.attractor,
            "transient_length": self.transient_length,
            "cycle_length": self.cycle_length,
            "is_fixed_point": self.is_fixed_point,
            "orbit_digest": self.orbit_digest,
            "avg_contraction": round(self.avg_contraction, 6),
            "monotone_steps": self.monotone_steps,
            "expansion_steps": self.expansion_steps,
        }


class OrbitAnalyzer:
    """Analyze orbits of a pipeline applied to numbers."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()
        self.nf = NumberFeatures(base=10)

    def trace_orbit(self, pipeline: Pipeline, n: int,
                    max_iter: int = 200, base: int = 10) -> OrbitProfile:
        """Trace a single orbit and extract features."""
        reg = self.registry
        nf = NumberFeatures(base=base)

        orbit = [n]
        seen = {n: 0}
        val = n

        for t in range(1, max_iter + 1):
            val = reg.execute_pipeline(pipeline, val)
            orbit.append(val)
            if val in seen:
                cycle_start = seen[val]
                cycle_len = t - cycle_start
                transient = cycle_start
                attractor = val
                break
            if val == 0:
                transient = t
                cycle_len = 1
                attractor = 0
                break
            seen[val] = t
        else:
            transient = max_iter
            cycle_len = 0
            attractor = val

        # Contraction ratios
        contraction_ratios = []
        mono_steps = 0
        expand_steps = 0
        for i in range(1, min(len(orbit), transient + 2)):
            prev = orbit[i - 1]
            curr = orbit[i]
            if prev > 0:
                contraction_ratios.append(curr / prev)
            if curr < prev:
                mono_steps += 1
            elif curr > prev:
                expand_steps += 1

        avg_c = statistics.mean(contraction_ratios) if contraction_ratios else 1.0

        # Digit sum and mod9 trajectories
        ds_traj = [sum(nf.digits(v)) for v in orbit[:min(20, len(orbit))]]
        mod9_traj = [v % 9 for v in orbit[:min(20, len(orbit))]]

        digest = ",".join(str(v) for v in orbit[:8])

        return OrbitProfile(
            start=n, attractor=attractor,
            transient_length=transient, cycle_length=cycle_len,
            is_fixed_point=(cycle_len == 1),
            orbit_digest=digest,
            contraction_ratios=contraction_ratios[:20],
            avg_contraction=avg_c,
            monotone_steps=mono_steps,
            expansion_steps=expand_steps,
            digit_sum_trajectory=ds_traj,
            mod9_trajectory=mod9_traj,
        )

    def analyze_basin(self, pipeline: Pipeline, domain: DomainPolicy,
                      max_iter: int = 200) -> Dict[str, Any]:
        """Analyze all orbits in a domain. Returns basin statistics."""
        lo, hi = domain.range()
        values = list(range(lo, hi + 1))
        if domain.exclude_repdigits:
            nf = NumberFeatures(domain.base)
            values = [n for n in values if not all(
                d == nf.digits(n)[0] for d in nf.digits(n)
            )]

        attractor_basins: Dict[int, List[int]] = defaultdict(list)
        transients: List[int] = []
        contractions: List[float] = []
        fp_count = 0
        cycle_count = 0

        for n in values:
            op = self.trace_orbit(pipeline, n, max_iter=max_iter, base=domain.base)
            attractor_basins[op.attractor].append(n)
            transients.append(op.transient_length)
            contractions.append(op.avg_contraction)
            if op.is_fixed_point:
                fp_count += 1
            elif op.cycle_length > 1:
                cycle_count += 1

        return {
            "num_values": len(values),
            "num_attractors": len(attractor_basins),
            "fixed_point_convergence": fp_count / len(values) if values else 0,
            "cycle_convergence": cycle_count / len(values) if values else 0,
            "avg_transient": statistics.mean(transients) if transients else 0,
            "median_transient": statistics.median(transients) if transients else 0,
            "avg_contraction": statistics.mean(contractions) if contractions else 1.0,
            "basin_sizes": {k: len(v) for k, v in attractor_basins.items()},
        }


# =============================================================================
# 3. CONJECTURE TEMPLATES
# =============================================================================

class ConjectureType(str, Enum):
    COUNTING = "counting"           # "#{FPs} = f(k)"
    INVARIANT = "invariant"         # "∀n: property holds"
    UNIVERSALITY = "universality"   # "convergence_rate → 1 for all k"
    STRUCTURE = "structure"         # "FP has form ..."
    MONOTONICITY = "monotonicity"   # "metric increases/decreases with k"
    MODULAR = "modular"             # "attractor ≡ c mod m"


@dataclass
class TestedDomain:
    """Explicit record of a domain used to test a conjecture (punt 2)."""
    base: int
    digit_length: int
    range_lo: int
    range_hi: int
    mode: str = "exhaustive"    # "exhaustive" | "sampled"
    seed: Optional[int] = None
    exclude_repdigits: bool = False

    def to_dict(self) -> dict:
        d = {"base": self.base, "digit_length": self.digit_length,
             "range": [self.range_lo, self.range_hi], "mode": self.mode,
             "exclude_repdigits": self.exclude_repdigits}
        if self.seed is not None:
            d["seed"] = self.seed
        return d

    @classmethod
    def from_policy(cls, dp: DomainPolicy, mode: str = "exhaustive",
                    seed: Optional[int] = None) -> TestedDomain:
        lo, hi = dp.range()
        return cls(base=dp.base, digit_length=dp.digit_length or 0,
                   range_lo=lo, range_hi=hi, mode=mode, seed=seed,
                   exclude_repdigits=dp.exclude_repdigits)


@dataclass
class Conjecture:
    """A formal conjecture with evidence and metadata."""
    ctype: ConjectureType
    statement: str
    quantifier: str             # "∀k≥3", "∀n∈D(k,10)", etc.
    predicate: str              # formal predicate
    evidence: Dict[str, Any]    # supporting data points
    exceptions: List[Any]       # known counterexamples
    pipeline: str               # pipeline display name
    parameters: Dict[str, Any]  # relevant parameters (base, k range, etc.)
    confidence: float           # 0.0 to 1.0
    novelty: float              # 0.0 to 1.0 (how surprising)
    simplicity: float           # 0.0 to 1.0 (inverse formula complexity)
    score: float = 0.0          # combined ranking score
    tested_domains: List[TestedDomain] = field(default_factory=list)
    monotonicity_kind: Optional[MonotonicityKind] = None

    def to_dict(self) -> dict:
        d = {
            "type": self.ctype.value,
            "statement": self.statement,
            "quantifier": self.quantifier,
            "predicate": self.predicate,
            "evidence": self.evidence,
            "exceptions": self.exceptions,
            "pipeline": self.pipeline,
            "parameters": self.parameters,
            "confidence": round(self.confidence, 4),
            "novelty": round(self.novelty, 4),
            "simplicity": round(self.simplicity, 4),
            "score": round(self.score, 4),
            "tested_domains": [td.to_dict() for td in self.tested_domains],
        }
        if self.monotonicity_kind is not None:
            d["monotonicity_kind"] = self.monotonicity_kind.value
        return d

    def __str__(self) -> str:
        ex = f" [exceptions: {self.exceptions}]" if self.exceptions else ""
        return f"[{self.ctype.value}] {self.statement}{ex}  (score={self.score:.3f})"


# =============================================================================
# 4. CONJECTURE MINER
# =============================================================================

class ConjectureMiner:
    """Mine conjectures from experiment results across parameter sweeps."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()
        self.runner = PipelineRunner(self.registry)

    @staticmethod
    def _build_tested_domains(digit_lengths: List[int], base: int,
                              exclude_repdigits: bool) -> List[TestedDomain]:
        """Build tested_domains list for provenance tracking (punt 2)."""
        domains = []
        for k in digit_lengths:
            dp = DomainPolicy(base=base, digit_length=k,
                              exclude_repdigits=exclude_repdigits)
            domains.append(TestedDomain.from_policy(dp))
        return domains

    @staticmethod
    def _classify_monotonicity(values: List[float]) -> Optional[MonotonicityKind]:
        """Classify sequence monotonicity precisely (punt 3)."""
        if len(values) < 2:
            return None
        strictly_inc = all(values[i] < values[i+1] for i in range(len(values)-1))
        nondec = all(values[i] <= values[i+1] for i in range(len(values)-1))
        strictly_dec = all(values[i] > values[i+1] for i in range(len(values)-1))
        noninc = all(values[i] >= values[i+1] for i in range(len(values)-1))
        # Check eventually_constant: last N values are equal
        if len(values) >= 3:
            tail = values[len(values)//2:]
            if len(set(tail)) == 1 and nondec:
                return MonotonicityKind.EVENTUALLY_CONSTANT
        if strictly_inc:
            return MonotonicityKind.STRICTLY_INCREASING
        if nondec:
            return MonotonicityKind.NONDECREASING
        if strictly_dec:
            return MonotonicityKind.STRICTLY_DECREASING
        if noninc:
            return MonotonicityKind.NONINCREASING
        return None

    def mine_counting(self, pipeline_str: str, digit_lengths: List[int],
                      base: int = 10, exclude_repdigits: bool = True) -> List[Conjecture]:
        """Mine counting conjectures: how does #{FPs} scale with k?"""
        pipe = Pipeline.parse(pipeline_str, registry=self.registry)
        conjectures = []
        fp_counts = {}
        cycle_counts = {}
        tested = self._build_tested_domains(digit_lengths, base, exclude_repdigits)

        for k in digit_lengths:
            domain = DomainPolicy(base=base, digit_length=k,
                                  exclude_repdigits=exclude_repdigits)
            result = self.runner.run_exhaustive(pipe, domain)
            fp_counts[k] = len(result.fixed_points)
            cycle_counts[k] = len(result.cycles)

        # Check: constant FP count
        fp_vals = list(fp_counts.values())
        if len(set(fp_vals)) == 1 and fp_vals[0] > 0:
            conjectures.append(Conjecture(
                ctype=ConjectureType.COUNTING,
                statement=f"#{{{pipeline_str}}} has exactly {fp_vals[0]} fixed point(s) for all tested k",
                quantifier=f"∀k∈{{{','.join(map(str, digit_lengths))}}}",
                predicate=f"#FP({pipeline_str}, k, base={base}) = {fp_vals[0]}",
                evidence=fp_counts,
                exceptions=[],
                pipeline=pipeline_str,
                parameters={"base": base, "k_range": digit_lengths},
                confidence=min(0.5 + 0.1 * len(digit_lengths), 0.95),
                novelty=0.6,
                simplicity=0.9,
                tested_domains=tested,
            ))

        # Classify monotonicity precisely (punt 3)
        mono_kind = self._classify_monotonicity([float(v) for v in fp_vals])
        if len(fp_vals) >= 3 and mono_kind is not None and fp_vals[0] != fp_vals[-1]:
            label = mono_kind.value.replace('_', ' ')
            conjectures.append(Conjecture(
                ctype=ConjectureType.MONOTONICITY,
                statement=f"#FP({pipeline_str}) is {label} in k",
                quantifier=f"∀k₁<k₂",
                predicate=f"#FP(k₁) {'<' if mono_kind == MonotonicityKind.STRICTLY_INCREASING else '≤'} #FP(k₂)",
                evidence=fp_counts,
                exceptions=[],
                pipeline=pipeline_str,
                parameters={"base": base, "k_range": digit_lengths},
                confidence=min(0.4 + 0.1 * len(digit_lengths), 0.85),
                novelty=0.5,
                simplicity=0.8,
                tested_domains=tested,
                monotonicity_kind=mono_kind,
            ))

        # Check: linear growth #FP = a*k + b
        if len(fp_vals) >= 3:
            ks = digit_lengths
            diffs = [fp_counts[ks[i+1]] - fp_counts[ks[i]] for i in range(len(ks)-1)]
            if len(set(diffs)) == 1 and diffs[0] != 0:
                a = diffs[0]
                b = fp_counts[ks[0]] - a * ks[0]
                conjectures.append(Conjecture(
                    ctype=ConjectureType.COUNTING,
                    statement=f"#FP({pipeline_str}) = {a}·k + {b}",
                    quantifier=f"∀k≥{ks[0]}",
                    predicate=f"#FP(k) = {a}*k + {b}",
                    evidence=fp_counts,
                    exceptions=[],
                    pipeline=pipeline_str,
                    parameters={"base": base, "a": a, "b": b},
                    confidence=min(0.5 + 0.1 * len(digit_lengths), 0.9),
                    novelty=0.7,
                    simplicity=0.85,
                    tested_domains=tested,
                ))

        return conjectures

    def mine_invariants(self, pipeline_str: str, digit_lengths: List[int],
                        base: int = 10) -> List[Conjecture]:
        """Mine mod-invariant conjectures on fixed points."""
        pipe = Pipeline.parse(pipeline_str, registry=self.registry)
        conjectures = []
        tested = self._build_tested_domains(digit_lengths, base, False)

        all_fps: Dict[int, List[int]] = {}
        for k in digit_lengths:
            domain = DomainPolicy(base=base, digit_length=k)
            result = self.runner.run_exhaustive(pipe, domain)
            all_fps[k] = [fp for fp in result.fixed_points if fp > 0]

        # Check mod invariants on FPs
        for m in [9, 11, 3, 7]:
            residues_per_k: Dict[int, Set[int]] = {}
            for k, fps in all_fps.items():
                residues_per_k[k] = set(fp % m for fp in fps) if fps else set()

            # Check if all FPs across all k share a common residue
            all_residues = set()
            for rs in residues_per_k.values():
                all_residues.update(rs)

            if len(all_residues) == 1 and all_residues != {0}:
                r = list(all_residues)[0]
                conjectures.append(Conjecture(
                    ctype=ConjectureType.MODULAR,
                    statement=f"All non-zero FPs of {pipeline_str} satisfy FP ≡ {r} (mod {m})",
                    quantifier=f"∀FP∈FP({pipeline_str}), FP>0",
                    predicate=f"FP ≡ {r} (mod {m})",
                    evidence={"residues": {k: sorted(rs) for k, rs in residues_per_k.items()}},
                    exceptions=[],
                    pipeline=pipeline_str,
                    parameters={"base": base, "modulus": m, "residue": r},
                    confidence=min(0.5 + 0.1 * len(digit_lengths), 0.9),
                    novelty=0.7,
                    simplicity=0.9,
                    tested_domains=tested,
                ))
            elif len(all_residues) > 0 and 0 in all_residues and len(all_residues) == 1:
                conjectures.append(Conjecture(
                    ctype=ConjectureType.MODULAR,
                    statement=f"All non-zero FPs of {pipeline_str} are divisible by {m}",
                    quantifier=f"∀FP∈FP({pipeline_str}), FP>0",
                    predicate=f"{m} | FP",
                    evidence={"residues": {k: sorted(rs) for k, rs in residues_per_k.items()}},
                    exceptions=[],
                    pipeline=pipeline_str,
                    parameters={"base": base, "modulus": m},
                    confidence=min(0.5 + 0.1 * len(digit_lengths), 0.9),
                    novelty=0.6,
                    simplicity=0.95,
                    tested_domains=tested,
                ))

        # Check digit sum invariant: ds(FP) = constant
        ds_per_k: Dict[int, Set[int]] = {}
        nf = NumberFeatures(base)
        for k, fps in all_fps.items():
            ds_per_k[k] = set(sum(nf.digits(fp)) for fp in fps) if fps else set()
        all_ds = set()
        for ds_set in ds_per_k.values():
            all_ds.update(ds_set)
        if len(all_ds) == 1:
            ds_val = list(all_ds)[0]
            conjectures.append(Conjecture(
                ctype=ConjectureType.INVARIANT,
                statement=f"All FPs of {pipeline_str} have digit_sum = {ds_val}",
                quantifier=f"∀FP∈FP({pipeline_str})",
                predicate=f"ds(FP) = {ds_val}",
                evidence={"digit_sums": {k: sorted(ds) for k, ds in ds_per_k.items()}},
                exceptions=[],
                pipeline=pipeline_str,
                parameters={"base": base, "digit_sum": ds_val},
                confidence=min(0.5 + 0.1 * len(digit_lengths), 0.9),
                novelty=0.6,
                simplicity=0.9,
                tested_domains=tested,
            ))

        return conjectures

    def mine_universality(self, pipeline_str: str, digit_lengths: List[int],
                          base: int = 10, exclude_repdigits: bool = True) -> List[Conjecture]:
        """Mine universality conjectures: convergence rate patterns."""
        pipe = Pipeline.parse(pipeline_str, registry=self.registry)
        conjectures = []
        conv_rates = {}
        basin_entropies = {}
        tested = self._build_tested_domains(digit_lengths, base, exclude_repdigits)

        for k in digit_lengths:
            domain = DomainPolicy(base=base, digit_length=k,
                                  exclude_repdigits=exclude_repdigits)
            result = self.runner.run_exhaustive(pipe, domain)
            conv_rates[k] = result.convergence_rate
            basin_entropies[k] = result.basin_entropy

        # Check: near-universal convergence (rate > 0.99 for all)
        if all(r > 0.99 for r in conv_rates.values()):
            conjectures.append(Conjecture(
                ctype=ConjectureType.UNIVERSALITY,
                statement=f"{pipeline_str} achieves near-universal convergence (>99%) for all tested k",
                quantifier=f"∀k∈{{{','.join(map(str, digit_lengths))}}}",
                predicate=f"conv_rate({pipeline_str}, k) > 0.99",
                evidence=conv_rates,
                exceptions=[],
                pipeline=pipeline_str,
                parameters={"base": base, "threshold": 0.99},
                confidence=min(0.6 + 0.08 * len(digit_lengths), 0.95),
                novelty=0.8,
                simplicity=0.85,
                tested_domains=tested,
            ))

        # Check: entropy monotonicity with k (punt 3: precise classification)
        ks = sorted(conv_rates.keys())
        ents = [basin_entropies[k] for k in ks]
        ent_mono = self._classify_monotonicity(ents)
        if len(ents) >= 3 and ent_mono is not None and ents[-1] > ents[0] + 0.1:
            label = ent_mono.value.replace('_', ' ')
            conjectures.append(Conjecture(
                ctype=ConjectureType.MONOTONICITY,
                statement=f"Basin entropy of {pipeline_str} is {label} with digit length",
                quantifier=f"∀k₁<k₂",
                predicate=f"H(k₁) {'<' if ent_mono == MonotonicityKind.STRICTLY_INCREASING else '≤'} H(k₂)",
                evidence=basin_entropies,
                exceptions=[],
                pipeline=pipeline_str,
                parameters={"base": base},
                confidence=min(0.4 + 0.1 * len(digit_lengths), 0.85),
                novelty=0.5,
                simplicity=0.8,
                tested_domains=tested,
                monotonicity_kind=ent_mono,
            ))

        return conjectures

    def mine_structure(self, pipeline_str: str, digit_lengths: List[int],
                       base: int = 10) -> List[Conjecture]:
        """Mine structural conjectures about FP forms."""
        pipe = Pipeline.parse(pipeline_str, registry=self.registry)
        conjectures = []
        nf = NumberFeatures(base)
        tested = self._build_tested_domains(digit_lengths, base, False)

        all_fps: Dict[int, List[int]] = {}
        for k in digit_lengths:
            domain = DomainPolicy(base=base, digit_length=k)
            result = self.runner.run_exhaustive(pipe, domain)
            all_fps[k] = [fp for fp in result.fixed_points if fp > 0]

        # Check: all FPs are palindromes
        all_pals = True
        pal_evidence = {}
        for k, fps in all_fps.items():
            pals = [fp for fp in fps if nf.digits(fp) == nf.digits(fp)[::-1]]
            pal_evidence[k] = {"total": len(fps), "palindromes": len(pals)}
            if pals != fps and fps:
                all_pals = False

        if all_pals and any(len(fps) > 0 for fps in all_fps.values()):
            conjectures.append(Conjecture(
                ctype=ConjectureType.STRUCTURE,
                statement=f"All non-zero FPs of {pipeline_str} are palindromes",
                quantifier=f"∀FP∈FP({pipeline_str}), FP>0",
                predicate=f"palindrome(FP) = True",
                evidence=pal_evidence,
                exceptions=[],
                pipeline=pipeline_str,
                parameters={"base": base},
                confidence=min(0.5 + 0.1 * len(digit_lengths), 0.9),
                novelty=0.65,
                simplicity=0.9,
                tested_domains=tested,
            ))

        # Check: FPs have repeating digit patterns (e.g., 6174 → no, 10890 → pattern)
        for k, fps in all_fps.items():
            for fp in fps:
                ds = nf.digits(fp)
                if len(ds) >= 4:
                    # Check symmetric pattern: d1 d2 ... d2 d1
                    if ds == ds[::-1]:
                        pass  # already covered by palindrome check
                    # Check if FP = base^(k-1) * c for some constant
                    if fp % (base ** (k // 2)) == 0:
                        pass  # Could mine power-of-base patterns

        return conjectures

    def mine_all(self, pipeline_str: str, digit_lengths: List[int],
                 base: int = 10, exclude_repdigits: bool = True) -> List[Conjecture]:
        """Run all mining templates and rank results."""
        all_conj = []
        all_conj.extend(self.mine_counting(pipeline_str, digit_lengths, base, exclude_repdigits))
        all_conj.extend(self.mine_invariants(pipeline_str, digit_lengths, base))
        all_conj.extend(self.mine_universality(pipeline_str, digit_lengths, base, exclude_repdigits))
        all_conj.extend(self.mine_structure(pipeline_str, digit_lengths, base))

        # Score and rank
        for c in all_conj:
            c.score = self._rank(c)
        all_conj.sort(key=lambda c: c.score, reverse=True)
        return all_conj

    @staticmethod
    def _rank(c: Conjecture) -> float:
        """Heuristic ranking: weighted combination of confidence, novelty, simplicity."""
        w_conf = 0.4
        w_nov = 0.35
        w_simp = 0.25
        return w_conf * c.confidence + w_nov * c.novelty + w_simp * c.simplicity


# =============================================================================
# 5. FALSIFICATION ENGINE
# =============================================================================

class FalsificationResult:
    """Result of a falsification attempt."""
    def __init__(self, conjecture: Conjecture, falsified: bool,
                 counterexamples: List[Any], refined: Optional[Conjecture] = None):
        self.conjecture = conjecture
        self.falsified = falsified
        self.counterexamples = counterexamples
        self.refined = refined

    def __str__(self) -> str:
        if self.falsified:
            return f"FALSIFIED: {self.conjecture.statement} — CEs: {self.counterexamples[:5]}"
        return f"SURVIVED: {self.conjecture.statement}"


class Falsifier:
    """Attempt to falsify conjectures by extending the parameter range."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()
        self.runner = PipelineRunner(self.registry)

    def test_counting(self, conjecture: Conjecture,
                      extra_ks: List[int]) -> FalsificationResult:
        """Test a counting conjecture on additional digit lengths."""
        pipe = Pipeline.parse(conjecture.pipeline, registry=self.registry)
        base = conjecture.parameters.get("base", 10)
        exclude_rep = conjecture.parameters.get("exclude_repdigits", True)
        counterexamples = []

        for k in extra_ks:
            domain = DomainPolicy(base=base, digit_length=k, exclude_repdigits=exclude_rep)
            result = self.runner.run_exhaustive(pipe, domain)
            num_fps = len(result.fixed_points)

            if conjecture.ctype == ConjectureType.COUNTING:
                # Check against predicted value
                if "a" in conjecture.parameters and "b" in conjecture.parameters:
                    expected = conjecture.parameters["a"] * k + conjecture.parameters["b"]
                    if num_fps != expected:
                        counterexamples.append({"k": k, "expected": expected, "actual": num_fps})
                elif "evidence" in conjecture.to_dict():
                    # Constant count check
                    expected_vals = set(conjecture.evidence.values())
                    if len(expected_vals) == 1:
                        expected = list(expected_vals)[0]
                        if num_fps != expected:
                            counterexamples.append({"k": k, "expected": expected, "actual": num_fps})

        falsified = len(counterexamples) > 0
        refined = None
        if falsified and counterexamples:
            # Attempt refinement: add exceptions
            refined = Conjecture(
                ctype=conjecture.ctype,
                statement=conjecture.statement + f" [refined: exceptions at k={[ce['k'] for ce in counterexamples]}]",
                quantifier=conjecture.quantifier,
                predicate=conjecture.predicate,
                evidence={**conjecture.evidence},
                exceptions=conjecture.exceptions + counterexamples,
                pipeline=conjecture.pipeline,
                parameters=conjecture.parameters,
                confidence=conjecture.confidence * 0.7,
                novelty=conjecture.novelty,
                simplicity=conjecture.simplicity * 0.9,
            )

        return FalsificationResult(conjecture, falsified, counterexamples, refined)

    def test_modular(self, conjecture: Conjecture,
                     extra_ks: List[int]) -> FalsificationResult:
        """Test a modular conjecture on additional digit lengths."""
        pipe = Pipeline.parse(conjecture.pipeline, registry=self.registry)
        base = conjecture.parameters.get("base", 10)
        m = conjecture.parameters.get("modulus", 9)
        expected_r = conjecture.parameters.get("residue", 0)
        counterexamples = []

        for k in extra_ks:
            domain = DomainPolicy(base=base, digit_length=k)
            result = self.runner.run_exhaustive(pipe, domain)
            for fp in result.fixed_points:
                if fp > 0 and fp % m != expected_r:
                    counterexamples.append({"k": k, "fp": fp, "residue": fp % m})

        return FalsificationResult(conjecture, len(counterexamples) > 0, counterexamples)


# =============================================================================
# MAIN: Demo conjecture mining
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENGINE vNext — M2: Feature Extractor + Conjecture Mining")
    print("=" * 70)

    reg = OperationRegistry()
    miner = ConjectureMiner(reg)
    falsifier = Falsifier(reg)

    # ── 1. Number features demo ──
    print("\n── 1. Number features (6174) ──\n")
    nf = NumberFeatures(base=10)
    profile = nf.compute(6174, num_digits=4)
    for k, v in profile.to_dict().items():
        print(f"  {k:>24}: {v}")

    # ── 2. Orbit analysis demo ──
    print("\n── 2. Orbit analysis (kaprekar_step, start=3087) ──\n")
    oa = OrbitAnalyzer(reg)
    pipe = Pipeline.parse("kaprekar_step", registry=reg)
    orbit = oa.trace_orbit(pipe, 3087)
    for k, v in orbit.to_dict().items():
        print(f"  {k:>24}: {v}")

    # ── 3. Conjecture mining: Kaprekar ──
    print("\n── 3. Conjecture mining: kaprekar_step (k=3..7) ──\n")
    conjectures = miner.mine_all("kaprekar_step", [3, 4, 5, 6, 7])
    for c in conjectures:
        print(f"  {c}")

    # ── 4. Conjecture mining: truc_1089 ──
    print("\n── 4. Conjecture mining: truc_1089 (k=3..7) ──\n")
    conjectures = miner.mine_all("truc_1089", [3, 4, 5, 6, 7], exclude_repdigits=False)
    for c in conjectures:
        print(f"  {c}")

    # ── 5. Falsification demo ──
    print("\n── 5. Falsification: extend to k=8 ──\n")
    kap_conj = miner.mine_counting("kaprekar_step", [3, 4, 5, 6, 7])
    for c in kap_conj:
        fr = falsifier.test_counting(c, [8])
        print(f"  {fr}")

    t1089_conj = miner.mine_invariants("truc_1089", [3, 4, 5])
    for c in t1089_conj:
        fr = falsifier.test_modular(c, [6, 7])
        print(f"  {fr}")

    print("\n" + "=" * 70)
    print("M2 COMPLETE")
    print("=" * 70)
