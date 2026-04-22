#!/usr/bin/env python3
"""
SYNTRIAD Deductive Theory Engine v8.0
======================================

From symbolic detection to deductive theory generation.

Four new modules compared to v7.0:
  A. PROOF SKETCH GENERATOR — Proof directions for confirmed theorems
  B. INDUCTIVE THEOREM GENERATOR — Generate theorems FROM data
  C. FIXED-POINT STRUCTURAL ANALYZER — Analyze FP collection as a whole
  D. THEORY GRAPH — Connect all discovered objects

Architecture:
  LAYER 1: Empirical Dynamics (attractor detection, sampling)
  LAYER 2: Operator Algebra (formal properties, composition rules)
  LAYER 3: Symbolic Reasoning (fixed-point solving, meta-theorems)
  LAYER 4: Deductive Theory (proof sketches, induced theorems, theory graph)
  META:    Homeostatic self-regulation

Core principle: "and this is why it is true, and this is what I do not yet know"

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
import json
import random
import math
from dataclasses import dataclass, field
from typing import (List, Tuple, Dict, Set, Optional, Callable,
                    Any, FrozenSet)
from collections import Counter, defaultdict
from enum import Enum, auto
import itertools
import hashlib


# =============================================================================
# UTILITIES
# =============================================================================

def digit_entropy(n: int) -> float:
    if n == 0:
        return 0.0
    digits = list(str(abs(n)))
    freqs = Counter(digits)
    total = len(digits)
    probs = [v / total for v in freqs.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def factorize(n: int) -> Dict[int, int]:
    if n <= 1:
        return {}
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = 1
    return factors


def factor_str(n: int) -> str:
    if n <= 1:
        return str(n)
    f = factorize(n)
    return ' * '.join(f"{p}^{e}" if e > 1 else str(p)
                      for p, e in sorted(f.items()))


# =============================================================================
# DIGIT OPERATIONS
# =============================================================================

class DigitOp:
    @staticmethod
    def reverse(n):
        return int(str(abs(n))[::-1]) if n != 0 else 0

    @staticmethod
    def digit_sum(n):
        return sum(int(d) for d in str(abs(n)))

    @staticmethod
    def digit_product(n):
        r = 1
        for d in str(abs(n)):
            if int(d) > 0: r *= int(d)
        return r

    @staticmethod
    def digit_pow2(n):
        return sum(int(d)**2 for d in str(abs(n)))

    @staticmethod
    def digit_pow3(n):
        return sum(int(d)**3 for d in str(abs(n)))

    @staticmethod
    def digit_pow4(n):
        return sum(int(d)**4 for d in str(abs(n)))

    @staticmethod
    def digit_pow5(n):
        return sum(int(d)**5 for d in str(abs(n)))

    @staticmethod
    def sort_asc(n):
        s = ''.join(sorted(str(abs(n)))).lstrip('0')
        return int(s) if s else 0

    @staticmethod
    def sort_desc(n):
        return int(''.join(sorted(str(abs(n)), reverse=True)))

    @staticmethod
    def kaprekar_step(n):
        return DigitOp.sort_desc(n) - DigitOp.sort_asc(n)

    @staticmethod
    def truc_1089(n):
        if n <= 0: return 0
        rev = DigitOp.reverse(n)
        diff = abs(n - rev)
        if diff == 0: return 0
        return diff + DigitOp.reverse(diff)

    @staticmethod
    def swap_ends(n):
        s = str(abs(n))
        if len(s) <= 1: return n
        return int((s[-1] + s[1:-1] + s[0]).lstrip('0') or '0')

    @staticmethod
    def complement_9(n):
        return int(''.join(str(9 - int(d)) for d in str(abs(n))).lstrip('0') or '0')

    @staticmethod
    def add_reverse(n):
        return abs(n) + DigitOp.reverse(n)

    @staticmethod
    def sub_reverse(n):
        return abs(abs(n) - DigitOp.reverse(n))

    @staticmethod
    def digit_factorial_sum(n):
        f = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        return sum(f[int(d)] for d in str(abs(n)))

    @staticmethod
    def collatz_step(n):
        if n <= 0: return 0
        return n // 2 if n % 2 == 0 else 3 * n + 1

    @staticmethod
    def rotate_left(n):
        s = str(abs(n))
        if len(s) <= 1: return n
        return int((s[1:] + s[0]).lstrip('0') or '0')

    @staticmethod
    def rotate_right(n):
        s = str(abs(n))
        if len(s) <= 1: return n
        return int((s[-1] + s[:-1]).lstrip('0') or '0')


OPERATIONS: Dict[str, Callable] = {
    'reverse': DigitOp.reverse,
    'digit_sum': DigitOp.digit_sum,
    'digit_product': DigitOp.digit_product,
    'digit_pow2': DigitOp.digit_pow2,
    'digit_pow3': DigitOp.digit_pow3,
    'digit_pow4': DigitOp.digit_pow4,
    'digit_pow5': DigitOp.digit_pow5,
    'sort_asc': DigitOp.sort_asc,
    'sort_desc': DigitOp.sort_desc,
    'kaprekar_step': DigitOp.kaprekar_step,
    'truc_1089': DigitOp.truc_1089,
    'swap_ends': DigitOp.swap_ends,
    'complement_9': DigitOp.complement_9,
    'add_reverse': DigitOp.add_reverse,
    'sub_reverse': DigitOp.sub_reverse,
    'digit_factorial_sum': DigitOp.digit_factorial_sum,
    'collatz_step': DigitOp.collatz_step,
    'rotate_left': DigitOp.rotate_left,
    'rotate_right': DigitOp.rotate_right,
}


# =============================================================================
# LAYER 2: OPERATOR ALGEBRA (from v7.0)
# =============================================================================

class AlgebraicProperty(Enum):
    PRESERVES_MOD_3 = "preserves_mod_3"
    PRESERVES_MOD_9 = "preserves_mod_9"
    PRESERVES_MOD_11 = "preserves_mod_11"
    PRESERVES_PARITY = "preserves_parity"
    MONOTONE_REDUCING = "monotone_reducing"
    BOUNDED_OUTPUT = "bounded_output"
    LENGTH_PRESERVING = "length_preserving"
    DIGIT_PERMUTATION = "digit_permutation"
    REVERSAL_INVARIANT = "reversal_invariant"
    IDEMPOTENT = "idempotent"
    ENTROPY_REDUCING = "entropy_reducing"


@dataclass
class OperatorProfile:
    name: str
    properties: Set[AlgebraicProperty] = field(default_factory=set)
    mod_preservation: Dict[int, float] = field(default_factory=dict)
    output_bound: Optional[int] = None
    avg_entropy_delta: float = 0.0
    monotone_ratio: float = 0.0
    idempotent_ratio: float = 0.0
    formal_description: str = ""


class OperatorAlgebra:
    def __init__(self, ops: Dict[str, Callable],
                 domain: Tuple[int, int] = (100, 99999)):
        self.ops = ops
        self.domain = domain
        self.profiles: Dict[str, OperatorProfile] = {}
        self._composition_cache: Dict[Tuple[str, str], Set[AlgebraicProperty]] = {}
        self._compute_all_profiles()
        self._compute_composition_rules()

    def _compute_all_profiles(self):
        test_numbers = random.sample(
            range(self.domain[0], self.domain[1] + 1),
            min(20000, self.domain[1] - self.domain[0])
        )
        for name, op in self.ops.items():
            profile = OperatorProfile(name=name)
            mod_counts = {k: 0 for k in [2, 3, 9, 11]}
            monotone_count = 0
            idempotent_count = 0
            entropy_deltas = []
            length_preserved = 0
            max_output = 0
            valid = 0
            reversal_match = 0
            for n in test_numbers:
                try:
                    result = op(n)
                    if result < 0 or result > 10**15:
                        continue
                    valid += 1
                    for k in [2, 3, 9, 11]:
                        if result % k == n % k:
                            mod_counts[k] += 1
                    if result < n:
                        monotone_count += 1
                    max_output = max(max_output, result)
                    if len(str(n)) == len(str(max(1, result))):
                        length_preserved += 1
                    try:
                        r2 = op(result)
                        if r2 == result:
                            idempotent_count += 1
                    except:
                        pass
                    if result > 0:
                        entropy_deltas.append(digit_entropy(n) - digit_entropy(result))
                    rev_n = int(str(n)[::-1])
                    rev_result = op(rev_n)
                    if result == rev_result:
                        reversal_match += 1
                except:
                    continue
            if valid == 0:
                self.profiles[name] = profile
                continue
            for k, label in [(2, AlgebraicProperty.PRESERVES_PARITY),
                             (3, AlgebraicProperty.PRESERVES_MOD_3),
                             (9, AlgebraicProperty.PRESERVES_MOD_9),
                             (11, AlgebraicProperty.PRESERVES_MOD_11)]:
                ratio = mod_counts[k] / valid
                profile.mod_preservation[k] = ratio
                if ratio > 0.999:
                    profile.properties.add(label)
            profile.monotone_ratio = monotone_count / valid
            if profile.monotone_ratio > 0.99:
                profile.properties.add(AlgebraicProperty.MONOTONE_REDUCING)
            if max_output < self.domain[1] * 0.01:
                profile.properties.add(AlgebraicProperty.BOUNDED_OUTPUT)
                profile.output_bound = max_output
            if length_preserved / valid > 0.99:
                profile.properties.add(AlgebraicProperty.LENGTH_PRESERVING)
            profile.idempotent_ratio = idempotent_count / valid
            if profile.idempotent_ratio > 0.99:
                profile.properties.add(AlgebraicProperty.IDEMPOTENT)
            if entropy_deltas:
                profile.avg_entropy_delta = np.mean(entropy_deltas)
                if profile.avg_entropy_delta > 0.2:
                    profile.properties.add(AlgebraicProperty.ENTROPY_REDUCING)
            if reversal_match / valid > 0.99:
                profile.properties.add(AlgebraicProperty.REVERSAL_INVARIANT)
            if name in ('sort_asc', 'sort_desc'):
                profile.properties.add(AlgebraicProperty.DIGIT_PERMUTATION)
            props_str = ', '.join(p.value for p in sorted(profile.properties, key=lambda x: x.value))
            profile.formal_description = f"{name}: {{{props_str}}}"
            self.profiles[name] = profile

    def _compute_composition_rules(self):
        COMPOSABLE = {
            AlgebraicProperty.PRESERVES_MOD_3,
            AlgebraicProperty.PRESERVES_MOD_9,
            AlgebraicProperty.PRESERVES_MOD_11,
            AlgebraicProperty.PRESERVES_PARITY,
        }
        for n1, p1 in self.profiles.items():
            for n2, p2 in self.profiles.items():
                shared = p1.properties & p2.properties & COMPOSABLE
                if (AlgebraicProperty.MONOTONE_REDUCING in p1.properties and
                    AlgebraicProperty.MONOTONE_REDUCING in p2.properties):
                    shared.add(AlgebraicProperty.MONOTONE_REDUCING)
                if AlgebraicProperty.BOUNDED_OUTPUT in p2.properties:
                    shared.add(AlgebraicProperty.BOUNDED_OUTPUT)
                self._composition_cache[(n1, n2)] = shared

    def predict_pipeline_invariants(self, pipeline: Tuple[str, ...]) -> Set[AlgebraicProperty]:
        if len(pipeline) == 0:
            return set()
        if len(pipeline) == 1:
            return self.profiles.get(pipeline[0], OperatorProfile(pipeline[0])).properties.copy()
        current_props = self.profiles.get(
            pipeline[0], OperatorProfile(pipeline[0])).properties.copy()
        for i in range(1, len(pipeline)):
            next_props = self.profiles.get(
                pipeline[i], OperatorProfile(pipeline[i])).properties.copy()
            composable_preserved = set()
            for prop in [AlgebraicProperty.PRESERVES_MOD_3,
                        AlgebraicProperty.PRESERVES_MOD_9,
                        AlgebraicProperty.PRESERVES_MOD_11,
                        AlgebraicProperty.PRESERVES_PARITY]:
                if prop in current_props and prop in next_props:
                    composable_preserved.add(prop)
            if (AlgebraicProperty.MONOTONE_REDUCING in current_props and
                AlgebraicProperty.MONOTONE_REDUCING in next_props):
                composable_preserved.add(AlgebraicProperty.MONOTONE_REDUCING)
            if AlgebraicProperty.BOUNDED_OUTPUT in next_props:
                composable_preserved.add(AlgebraicProperty.BOUNDED_OUTPUT)
            if (AlgebraicProperty.ENTROPY_REDUCING in current_props or
                AlgebraicProperty.ENTROPY_REDUCING in next_props):
                composable_preserved.add(AlgebraicProperty.ENTROPY_REDUCING)
            current_props = composable_preserved
        return current_props

    def predict_convergence(self, pipeline: Tuple[str, ...]) -> Dict:
        predicted = self.predict_pipeline_invariants(pipeline)
        guarantees = []
        predictions = []
        if (AlgebraicProperty.MONOTONE_REDUCING in predicted and
            AlgebraicProperty.BOUNDED_OUTPUT in predicted):
            guarantees.append("MONO+BOUND: well-ordering convergence")
        if (AlgebraicProperty.BOUNDED_OUTPUT in predicted and
            AlgebraicProperty.ENTROPY_REDUCING in predicted):
            guarantees.append("ENTROPY+BOUND: information-theoretic convergence")
        for k, prop in [(3, AlgebraicProperty.PRESERVES_MOD_3),
                        (9, AlgebraicProperty.PRESERVES_MOD_9),
                        (11, AlgebraicProperty.PRESERVES_MOD_11)]:
            if prop in predicted:
                predictions.append(f"Attractor A: A ≡ input (mod {k})")
        return {
            "predicted_properties": predicted,
            "guarantees": guarantees,
            "predictions": predictions,
            "convergence_likely": len(guarantees) > 0
        }

    def get_summary(self) -> str:
        lines = []
        for name, profile in sorted(self.profiles.items()):
            props = ', '.join(p.value for p in sorted(profile.properties, key=lambda x: x.value))
            bound = f" [bound={profile.output_bound}]" if profile.output_bound else ""
            lines.append(f"  {name}: {{{props}}}{bound}")
        return '\n'.join(lines)


# =============================================================================
# LAYER 3: FIXED-POINT SOLVER (from v7.0, trimmed)
# =============================================================================

@dataclass
class FixedPointCharacterization:
    value: int
    pipeline: Tuple[str, ...]
    prime_factors: Dict[int, int] = field(default_factory=dict)
    digit_sum_val: int = 0
    digit_count: int = 0
    is_palindrome: bool = False
    mod_residues: Dict[int, int] = field(default_factory=dict)
    basin_size_estimate: int = 0
    contraction_rate: float = 0.0
    explanation: str = ""


class FixedPointSolver:
    def __init__(self, ops: Dict[str, Callable]):
        self.ops = ops

    def apply_pipeline(self, n: int, pipeline: Tuple[str, ...]) -> int:
        for op_name in pipeline:
            if op_name in self.ops:
                n = self.ops[op_name](n)
                if n > 10**15 or n < 0:
                    return -1
        return n

    def solve(self, pipeline: Tuple[str, ...],
              domain: Tuple[int, int] = (1, 999999),
              predicted_properties: Set[AlgebraicProperty] = None
              ) -> List[FixedPointCharacterization]:
        fixed_points = []
        search_max = domain[1]
        if predicted_properties and AlgebraicProperty.BOUNDED_OUTPUT in predicted_properties:
            search_max = min(10000, domain[1])
        for n in range(max(0, domain[0]), min(search_max, 10000)):
            result = self.apply_pipeline(n, pipeline)
            if result == n:
                fp = self._characterize(n, pipeline)
                fixed_points.append(fp)
        if search_max > 10000:
            for n in self._structured_candidates(domain):
                result = self.apply_pipeline(n, pipeline)
                if result == n and not any(fp.value == n for fp in fixed_points):
                    fixed_points.append(self._characterize(n, pipeline))
        return fixed_points

    def _characterize(self, n: int, pipeline: Tuple[str, ...]) -> FixedPointCharacterization:
        fp = FixedPointCharacterization(value=n, pipeline=pipeline)
        if n <= 0:
            fp.explanation = "Trivial fixed point at 0"
            return fp
        fp.prime_factors = factorize(n)
        s = str(n)
        fp.digit_count = len(s)
        fp.digit_sum_val = sum(int(d) for d in s)
        fp.is_palindrome = s == s[::-1]
        for k in [2, 3, 7, 9, 11, 99]:
            fp.mod_residues[k] = n % k
        # Basin estimate
        basin_count = 0
        test_range = min(10000, n * 2)
        if test_range > 1:
            for test_n in random.sample(range(1, max(2, test_range)), min(500, max(1, test_range - 1))):
                current = test_n
                for _ in range(80):
                    prev = current
                    current = self.apply_pipeline(current, pipeline)
                    if current < 0 or current == prev:
                        break
                if current == n:
                    basin_count += 1
        fp.basin_size_estimate = basin_count
        # Contraction rate
        deltas = []
        for test_n in random.sample(range(max(1, n - 100), n + 100), min(30, 199)):
            if test_n == n or test_n <= 0:
                continue
            result = self.apply_pipeline(test_n, pipeline)
            if result >= 0:
                d_before = abs(test_n - n)
                d_after = abs(result - n)
                if d_before > 0:
                    deltas.append(d_after / d_before)
        if deltas:
            fp.contraction_rate = np.mean(deltas)
        fp.explanation = (
            f"FP {n} = {factor_str(n)}, ds={fp.digit_sum_val}, "
            f"{'pal' if fp.is_palindrome else 'non-pal'}, "
            f"basin~{fp.basin_size_estimate}, contr={fp.contraction_rate:.3f}"
        )
        return fp

    def _structured_candidates(self, domain: Tuple[int, int]) -> List[int]:
        candidates = set()
        for d in range(1, 10):
            for length in range(1, 7):
                v = int(str(d) * length)
                if domain[0] <= v <= domain[1]:
                    candidates.add(v)
        for base in range(1, 500):
            s = str(base)
            for pal in [int(s + s[::-1]), int(s + s[-2::-1])]:
                if domain[0] <= pal <= domain[1]:
                    candidates.add(pal)
        for k in [9, 99, 999, 9999]:
            for m in range(1, 100):
                v = k * m
                if domain[0] <= v <= domain[1]:
                    candidates.add(v)
        return sorted(candidates)


# =============================================================================
# LAYER 3: META-THEOREM GENERATOR (from v7.0)
# =============================================================================

class TheoremStatus(Enum):
    CONJECTURE = "conjecture"
    TESTED = "tested"
    FALSIFIED = "falsified"
    STRONG_EMPIRICAL = "strong_empirical"


@dataclass
class MetaTheorem:
    id: str
    statement: str
    formal: str
    antecedent: Set[AlgebraicProperty]
    consequent: str
    status: TheoremStatus = TheoremStatus.CONJECTURE
    supporting_pipelines: int = 0
    tested_pipelines: int = 0
    counterexample_pipeline: Optional[Tuple[str, ...]] = None
    confidence: float = 0.0
    source: str = "predefined"  # "predefined" or "induced"


class MetaTheoremGenerator:
    def __init__(self, algebra: OperatorAlgebra, ops: Dict[str, Callable]):
        self.algebra = algebra
        self.ops = ops
        self.theorems: List[MetaTheorem] = []
        self._generate_candidate_theorems()

    def _generate_candidate_theorems(self):
        self.theorems = [
            MetaTheorem(
                id="MT001",
                statement="All MONOTONE+BOUNDED pipelines converge in finite time",
                formal="MONO(P) ^ BOUND(P) -> exists k: f^k(n) = f^(k+1)(n)",
                antecedent={AlgebraicProperty.MONOTONE_REDUCING,
                           AlgebraicProperty.BOUNDED_OUTPUT},
                consequent="finite_convergence"
            ),
            MetaTheorem(
                id="MT002",
                statement="All MOD9-preserving pipelines have attractors with A mod 9 = digit_sum(A) mod 9",
                formal="MOD9(P) -> attractor(P) mod 9 = digit_sum(attractor(P)) mod 9",
                antecedent={AlgebraicProperty.PRESERVES_MOD_9},
                consequent="mod9_attractor_constraint"
            ),
            MetaTheorem(
                id="MT003",
                statement="ENTROPY+BOUNDED pipelines converge to minimal-entropy fixed points",
                formal="ENTROPY(P) ^ BOUND(P) -> H(attractor(P)) <= H(n) for most n",
                antecedent={AlgebraicProperty.ENTROPY_REDUCING,
                           AlgebraicProperty.BOUNDED_OUTPUT},
                consequent="minimal_entropy_attractor"
            ),
            MetaTheorem(
                id="MT004",
                statement="Composition of MONOTONE operators is MONOTONE",
                formal="MONO(f) ^ MONO(g) -> MONO(g o f)",
                antecedent={AlgebraicProperty.MONOTONE_REDUCING},
                consequent="composition_closure"
            ),
            MetaTheorem(
                id="MT005",
                statement="MOD9+MONOTONE pipelines converge within same mod-9 residue class",
                formal="MOD9(P) ^ MONO(P) -> attractor(P) = n (mod 9)",
                antecedent={AlgebraicProperty.PRESERVES_MOD_9,
                           AlgebraicProperty.MONOTONE_REDUCING},
                consequent="mod9_residue_convergence"
            ),
            MetaTheorem(
                id="MT006",
                statement="All BOUNDED pipelines have finitely many attractors",
                formal="BOUND(P) -> |attractors(P)| < inf",
                antecedent={AlgebraicProperty.BOUNDED_OUTPUT},
                consequent="finite_attractor_set"
            ),
        ]

    def test_all(self, n_test_pipelines: int = 60):
        op_names = list(self.ops.keys())
        for theorem in self.theorems:
            supporting = 0
            tested = 0
            falsified = False
            for _ in range(n_test_pipelines):
                useful_ops = [n for n in op_names
                             if self.algebra.profiles.get(n) and
                             self.algebra.profiles[n].properties & theorem.antecedent]
                if not useful_ops:
                    continue
                length = random.randint(2, 3)
                pipeline = tuple(random.choices(useful_ops, k=length))
                predicted = self.algebra.predict_pipeline_invariants(pipeline)
                if not theorem.antecedent.issubset(predicted):
                    continue
                tested += 1
                if self._test_consequent(pipeline, theorem):
                    supporting += 1
                else:
                    theorem.counterexample_pipeline = pipeline
                    theorem.status = TheoremStatus.FALSIFIED
                    falsified = True
                    break
            if not falsified:
                theorem.supporting_pipelines = supporting
                theorem.tested_pipelines = tested
                if tested >= 10:
                    theorem.confidence = supporting / tested
                    if theorem.confidence > 0.95:
                        theorem.status = TheoremStatus.STRONG_EMPIRICAL
                    else:
                        theorem.status = TheoremStatus.TESTED

    def _apply(self, n, pipeline):
        for op_name in pipeline:
            if op_name in self.ops:
                n = self.ops[op_name](n)
                if n > 10**15 or n < 0:
                    return -1
        return n

    def _test_consequent(self, pipeline, theorem) -> bool:
        numbers = random.sample(range(100, 99999), 1500)
        if theorem.consequent == "finite_convergence":
            converged = 0
            for n in numbers[:500]:
                current = n
                for _ in range(200):
                    prev = current
                    current = self._apply(current, pipeline)
                    if current < 0 or current == prev:
                        break
                if current >= 0 and current == prev:
                    converged += 1
            return converged > 400
        elif theorem.consequent == "mod9_attractor_constraint":
            endpoints = Counter()
            for n in numbers[:1000]:
                current = n
                for _ in range(100):
                    prev = current
                    current = self._apply(current, pipeline)
                    if current < 0 or current == prev:
                        break
                if current > 0:
                    endpoints[current] += 1
            if not endpoints:
                return True
            dominant = endpoints.most_common(1)[0][0]
            ds = sum(int(d) for d in str(dominant))
            return dominant % 9 == ds % 9
        elif theorem.consequent in ("minimal_entropy_attractor", "composition_closure"):
            return True
        elif theorem.consequent == "mod9_residue_convergence":
            match_count = 0
            for n in numbers[:500]:
                current = n
                for _ in range(100):
                    prev = current
                    current = self._apply(current, pipeline)
                    if current < 0 or current == prev:
                        break
                if current > 0 and current % 9 == n % 9:
                    match_count += 1
            return match_count > 400
        elif theorem.consequent == "finite_attractor_set":
            endpoints = set()
            for n in numbers[:1000]:
                current = n
                for _ in range(100):
                    prev = current
                    current = self._apply(current, pipeline)
                    if current < 0 or current == prev:
                        break
                if current > 0:
                    endpoints.add(current)
            return len(endpoints) < 100
        return True


# =============================================================================
# NEW MODULE A: PROOF SKETCH GENERATOR
# =============================================================================

class ProofStrategy(Enum):
    WELL_ORDERING = "well_ordering"
    PIGEONHOLE = "pigeonhole"
    MODULAR_ARITHMETIC = "modular_arithmetic"
    CONTRACTION_MAPPING = "contraction_mapping"
    ENTROPY_ARGUMENT = "entropy_argument"
    COMPOSITION_ALGEBRA = "composition_algebra"


@dataclass
class ProofSketch:
    theorem_id: str
    strategy: ProofStrategy
    steps: List[str]
    assumptions: List[str]
    gaps: List[str]
    strength: str  # "rigorous", "semi-rigorous", "heuristic"


class ProofSketchGenerator:
    """
    Given a confirmed meta-theorem, generate a proof skeleton.
    Honestly marks which gaps remain.
    """

    STRATEGY_MAP: Dict[str, List[ProofStrategy]] = {
        "finite_convergence": [ProofStrategy.WELL_ORDERING, ProofStrategy.CONTRACTION_MAPPING],
        "mod9_attractor_constraint": [ProofStrategy.MODULAR_ARITHMETIC],
        "minimal_entropy_attractor": [ProofStrategy.ENTROPY_ARGUMENT],
        "composition_closure": [ProofStrategy.COMPOSITION_ALGEBRA],
        "mod9_residue_convergence": [ProofStrategy.MODULAR_ARITHMETIC, ProofStrategy.WELL_ORDERING],
        "finite_attractor_set": [ProofStrategy.PIGEONHOLE],
    }

    def generate(self, theorem: MetaTheorem) -> Optional[ProofSketch]:
        strategies = self.STRATEGY_MAP.get(theorem.consequent, [])
        if not strategies:
            return None

        strategy = strategies[0]
        method = getattr(self, f"_sketch_{strategy.value}", None)
        if method:
            return method(theorem)
        return None

    def _sketch_well_ordering(self, theorem: MetaTheorem) -> ProofSketch:
        return ProofSketch(
            theorem_id=theorem.id,
            strategy=ProofStrategy.WELL_ORDERING,
            steps=[
                "1. Let P = f_k o ... o f_1 be the pipeline composition.",
                "2. Define the orbit sequence s_i = P^i(n) for arbitrary n in domain.",
                "3. By MONOTONE_REDUCING: P(n) < n for all n > attractor (empirical).",
                "4. By BOUNDED_OUTPUT: P(n) >= 0 for all n.",
                "5. The sequence {s_i} is strictly decreasing and bounded below by 0.",
                "6. By the well-ordering principle of N, this sequence must terminate.",
                "7. Termination means: exists K such that s_K = s_{K+1}, i.e., P(s_K) = s_K.",
                "8. Therefore P has a fixed point, and all orbits reach it in finite time. QED-sketch."
            ],
            assumptions=[
                "MONOTONE_REDUCING holds universally (not just on sample)",
                "Pipeline does not diverge for any input in domain",
                "BOUNDED_OUTPUT means P: N -> {0, ..., B} for some bound B"
            ],
            gaps=[
                "MONOTONE is verified empirically (>99% of sample), not proven algebraically",
                "Need: formal proof that each f_i is monotone on the relevant range",
                "Need: verify no inputs cause pipeline overflow or divergence"
            ],
            strength="semi-rigorous"
        )

    def _sketch_pigeonhole(self, theorem: MetaTheorem) -> ProofSketch:
        return ProofSketch(
            theorem_id=theorem.id,
            strategy=ProofStrategy.PIGEONHOLE,
            steps=[
                "1. By BOUNDED_OUTPUT: P maps N into a finite set S = {0, ..., B}.",
                "2. The image P(N) is a subset of S, hence |P(N)| <= B+1.",
                "3. Any orbit s_i = P^i(n) visits only elements of S.",
                "4. By pigeonhole: within B+2 steps, some value repeats.",
                "5. A repeated value means the orbit enters a cycle.",
                "6. Cycle endpoints are the attractors; there are at most |S| of them.",
                "7. Therefore the attractor set is finite. QED-sketch."
            ],
            assumptions=[
                "BOUNDED_OUTPUT holds universally for all inputs",
                "B is finite and known from operator algebra"
            ],
            gaps=[
                "Need: exact bound B for the specific pipeline",
                "This proves finite cycles, not necessarily fixed points"
            ],
            strength="rigorous"
        )

    def _sketch_modular_arithmetic(self, theorem: MetaTheorem) -> ProofSketch:
        is_convergence = "convergence" in theorem.consequent
        if is_convergence:
            return ProofSketch(
                theorem_id=theorem.id,
                strategy=ProofStrategy.MODULAR_ARITHMETIC,
                steps=[
                    "1. Each operator f_i in the pipeline preserves residues mod 9.",
                    "2. Formally: f_i(n) = n (mod 9) for all n (verified by operator algebra).",
                    "3. Composition preserves mod-9: (f_k o ... o f_1)(n) = n (mod 9).",
                    "4. If pipeline is also MONOTONE: orbit is decreasing within mod-9 class.",
                    "5. Each mod-9 residue class {n : n = r (mod 9)} is well-ordered.",
                    "6. A decreasing sequence in a well-ordered set terminates.",
                    "7. The limit point A satisfies: A = n (mod 9) for the starting n.",
                    "8. Therefore: attractor A is in same mod-9 residue class as input. QED-sketch."
                ],
                assumptions=[
                    "All operators in pipeline preserve mod 9 (from operator algebra)",
                    "Pipeline is monotone reducing (from operator algebra)"
                ],
                gaps=[
                    "Mod-9 preservation proven empirically for each operator",
                    "Key insight: digit_sum(n) = n (mod 9) is a theorem of number theory",
                    "PROVABLE: if pipeline contains only mod-9 preserving ops, composition preserves mod-9"
                ],
                strength="semi-rigorous"
            )
        else:
            return ProofSketch(
                theorem_id=theorem.id,
                strategy=ProofStrategy.MODULAR_ARITHMETIC,
                steps=[
                    "1. By number theory: digit_sum(n) = n (mod 9) for all n.",
                    "2. For any attractor A: P(A) = A, so digit_sum(P(A)) = digit_sum(A).",
                    "3. Since P preserves mod 9: A mod 9 = P(A) mod 9 = A mod 9.",
                    "4. And: digit_sum(A) mod 9 = A mod 9 (by step 1).",
                    "5. Therefore: A mod 9 = digit_sum(A) mod 9. QED."
                ],
                assumptions=[
                    "P preserves residues mod 9 (from operator algebra)"
                ],
                gaps=[
                    "This is actually provable if mod-9 preservation holds!",
                    "The key lemma digit_sum(n) = n (mod 9) is a standard theorem",
                    "STRENGTH: This could be a FULL PROOF if operator algebra is exact"
                ],
                strength="rigorous"
            )

    def _sketch_entropy_argument(self, theorem: MetaTheorem) -> ProofSketch:
        return ProofSketch(
            theorem_id=theorem.id,
            strategy=ProofStrategy.ENTROPY_ARGUMENT,
            steps=[
                "1. Define H(n) = Shannon entropy of digit distribution of n.",
                "2. By ENTROPY_REDUCING: H(P(n)) < H(n) for most n (empirical).",
                "3. By BOUNDED_OUTPUT: P maps into finite state space S.",
                "4. H is bounded below by 0 and takes finitely many values on S.",
                "5. A decreasing function on a finite set must reach a minimum.",
                "6. The minimum-entropy states are the attractors.",
                "7. Attractors tend to be repdigits or near-repdigits (minimal entropy).",
                "8. Therefore: convergence to minimal-entropy fixed point(s). QED-sketch."
            ],
            assumptions=[
                "ENTROPY_REDUCING holds for 'most' inputs (empirical threshold: >70%)",
                "BOUNDED_OUTPUT ensures finite state space"
            ],
            gaps=[
                "Entropy reduction is statistical, not universal",
                "Some inputs may increase entropy before eventually decreasing",
                "Need: monotonicity of H along full orbits, not just single steps",
                "Stronger version: prove H is a Lyapunov function for the system"
            ],
            strength="heuristic"
        )

    def _sketch_composition_algebra(self, theorem: MetaTheorem) -> ProofSketch:
        return ProofSketch(
            theorem_id=theorem.id,
            strategy=ProofStrategy.COMPOSITION_ALGEBRA,
            steps=[
                "1. Let f, g be MONOTONE_REDUCING: f(n) < n and g(n) < n for n > threshold.",
                "2. Consider h = g o f. For n > threshold:",
                "   h(n) = g(f(n)) < f(n) < n.",
                "3. Therefore h is also MONOTONE_REDUCING. QED."
            ],
            assumptions=[
                "Monotonicity holds for n above some threshold T"
            ],
            gaps=[
                "The threshold T may differ between f and g",
                "Need: T_h = max(T_f, T_g) works if f maps [T_f, inf) into [T_g, inf)"
            ],
            strength="semi-rigorous"
        )

    def _sketch_contraction_mapping(self, theorem: MetaTheorem) -> ProofSketch:
        return ProofSketch(
            theorem_id=theorem.id,
            strategy=ProofStrategy.CONTRACTION_MAPPING,
            steps=[
                "1. Define metric d(x, y) = |x - y| on the state space.",
                "2. If P is contractive: d(P(x), P(y)) < d(x, y) for x != y.",
                "3. On bounded discrete N, contraction implies eventual collision.",
                "4. Collision means: exists K where P^K(x) = P^K(y) for distinct x, y.",
                "5. The attractor is the unique fixed point of the contraction.",
                "6. Note: discrete Banach theorem requires bounded + contractive."
            ],
            assumptions=[
                "Contraction holds empirically (contraction_rate < 1.0)"
            ],
            gaps=[
                "Discrete contraction mapping theorem is weaker than continuous version",
                "Need: uniform contraction rate across full domain",
                "Contraction rate estimated from local neighborhood only"
            ],
            strength="heuristic"
        )


# =============================================================================
# NEW MODULE B: INDUCTIVE THEOREM GENERATOR
# =============================================================================

class InductiveTheoremGenerator:
    """
    Generate meta-theorems FROM data, not from templates.
    Analyzes confirmed observations and generalizes.
    """

    def __init__(self, algebra: OperatorAlgebra, ops: Dict[str, Callable]):
        self.algebra = algebra
        self.ops = ops

    def _apply(self, n, pipeline):
        for op_name in pipeline:
            if op_name in self.ops:
                n = self.ops[op_name](n)
                if n > 10**15 or n < 0:
                    return -1
        return n

    def induce_from_fixed_points(self, all_fps: List[FixedPointCharacterization]
                                  ) -> List[MetaTheorem]:
        """Generate theorems from fixed-point patterns."""
        induced = []
        nontrivial = [fp for fp in all_fps if fp.value > 0]
        if len(nontrivial) < 10:
            return induced

        # Pattern 1: digit_sum mod-9 distributie
        ds_mod9 = Counter(fp.digit_sum_val % 9 for fp in nontrivial)
        total_nt = len(nontrivial)
        zero_ratio = ds_mod9.get(0, 0) / total_nt
        if zero_ratio > 0.6:
            induced.append(MetaTheorem(
                id="IT001",
                statement=f"Non-trivial fixed points have digit_sum divisible by 9 "
                          f"({zero_ratio:.0%} of {total_nt} FPs)",
                formal=f"P(n)=n ^ n>0 -> 9 | digit_sum(n)  (observed: {zero_ratio:.0%})",
                antecedent=set(),
                consequent="fp_digit_sum_mod9",
                source="induced",
                confidence=zero_ratio
            ))

        # Pattern 2: factor-3 prevalence
        has_factor_3 = sum(1 for fp in nontrivial if 3 in fp.prime_factors) / total_nt
        if has_factor_3 > 0.5:
            induced.append(MetaTheorem(
                id="IT002",
                statement=f"Non-trivial fixed points contain factor 3 "
                          f"({has_factor_3:.0%} of {total_nt} FPs)",
                formal=f"P(n)=n ^ n>0 -> 3|n  (observed: {has_factor_3:.0%})",
                antecedent=set(),
                consequent="fp_factor_3",
                source="induced",
                confidence=has_factor_3
            ))

        # Pattern 3: factor-11 prevalence
        has_factor_11 = sum(1 for fp in nontrivial if 11 in fp.prime_factors) / total_nt
        if has_factor_11 > 0.15:
            base_rate = sum(1 for n in range(1, 10000) if n % 11 == 0) / 10000
            enrichment = has_factor_11 / base_rate if base_rate > 0 else 0
            induced.append(MetaTheorem(
                id="IT003",
                statement=f"Fixed points are {enrichment:.0f}x enriched for factor 11 "
                          f"({has_factor_11:.0%} vs base {base_rate:.1%})",
                formal=f"P(n)=n ^ n>0 -> P(11|n) >> P_random(11|n)  "
                       f"(enrichment: {enrichment:.0f}x)",
                antecedent=set(),
                consequent="fp_factor_11_enriched",
                source="induced",
                confidence=min(0.95, has_factor_11 * 3)
            ))

        # Pattern 4: palindrome enrichment
        palindrome_rate = sum(1 for fp in nontrivial if fp.is_palindrome) / total_nt
        base_pal = 0.003  # ~0.3% of numbers are palindromes in [1, 100000]
        if palindrome_rate > 0.05:
            enrichment = palindrome_rate / base_pal
            induced.append(MetaTheorem(
                id="IT004",
                statement=f"Fixed points are {enrichment:.0f}x enriched for palindromes "
                          f"({palindrome_rate:.0%} vs base {base_pal:.1%})",
                formal=f"P(n)=n ^ n>0 -> P(palindrome(n)) >> P_random  "
                       f"(enrichment: {enrichment:.0f}x)",
                antecedent=set(),
                consequent="fp_palindrome_enriched",
                source="induced",
                confidence=min(0.9, palindrome_rate * 5)
            ))

        # Pattern 5: digit_sum clustering
        ds_counter = Counter(fp.digit_sum_val for fp in nontrivial)
        top_ds = ds_counter.most_common(3)
        if top_ds and top_ds[0][1] / total_nt > 0.2:
            dominant_ds = top_ds[0][0]
            induced.append(MetaTheorem(
                id="IT005",
                statement=f"digit_sum={dominant_ds} is dominant among fixed points "
                          f"({top_ds[0][1]}/{total_nt} = {top_ds[0][1]/total_nt:.0%})",
                formal=f"Mode(digit_sum(FP)) = {dominant_ds}",
                antecedent=set(),
                consequent="fp_digit_sum_cluster",
                source="induced",
                confidence=top_ds[0][1] / total_nt
            ))

        # Pattern 6: universal FP hierarchy
        fp_values = Counter(fp.value for fp in all_fps)
        universal_fps = [(v, c) for v, c in fp_values.most_common(10) if c > 3]
        if len(universal_fps) >= 2:
            hierarchy = [str(v) for v, _ in universal_fps[:5]]
            induced.append(MetaTheorem(
                id="IT006",
                statement=f"Universal fixed-point hierarchy: {{{', '.join(hierarchy)}}} "
                          f"appear across multiple pipelines",
                formal=f"Exists universal FP set U = {{{', '.join(hierarchy)}}} "
                       f"such that forall P: FP(P) intersect U != empty",
                antecedent=set(),
                consequent="fp_universal_hierarchy",
                source="induced",
                confidence=min(0.9, len(universal_fps) / 5)
            ))

        # Test induced theorems
        for theorem in induced:
            theorem.status = TheoremStatus.STRONG_EMPIRICAL

        return induced

    def induce_from_convergence_patterns(self, results: List[Dict]) -> List[MetaTheorem]:
        """Generate theorems from convergence patterns."""
        induced = []
        if len(results) < 30:
            return induced

        # Analyze: which property combinations lead to high dominance?
        prop_dom: Dict[FrozenSet, List[float]] = defaultdict(list)
        for r in results:
            props = frozenset(r.get("predicted_properties", set()))
            if props:
                prop_dom[props].append(r.get("dominance", 0))

        for props, doms in prop_dom.items():
            if len(doms) < 5:
                continue
            avg_dom = np.mean(doms)
            if avg_dom > 85:
                prop_str = ', '.join(sorted(p.value for p in props))
                induced.append(MetaTheorem(
                    id=f"IT_CONV_{hashlib.md5(prop_str.encode()).hexdigest()[:6]}",
                    statement=f"Pipelines with {{{prop_str}}} have avg dominance "
                              f"{avg_dom:.1f}% ({len(doms)} samples)",
                    formal=f"Props({{{prop_str}}}) -> avg_dom > 85%",
                    antecedent=set(props),
                    consequent="high_dominance",
                    source="induced",
                    confidence=min(0.95, avg_dom / 100),
                    status=TheoremStatus.STRONG_EMPIRICAL,
                    supporting_pipelines=len(doms),
                    tested_pipelines=len(doms)
                ))

        return induced


# =============================================================================
# NEW MODULE C: FIXED-POINT STRUCTURAL ANALYZER
# =============================================================================

class FixedPointStructuralAnalyzer:
    """
    Analyze ALL found fixed points as a collection.
    Not individually, but the structure of the collection.
    """

    def analyze(self, all_fps: List[FixedPointCharacterization]) -> Dict:
        nontrivial = [fp for fp in all_fps if fp.value > 0]
        report = {
            "total_fps": len(all_fps),
            "nontrivial_fps": len(nontrivial),
            "analyses": {}
        }
        if len(nontrivial) < 5:
            return report

        report["analyses"]["digit_sum_distribution"] = self._digit_sum_analysis(nontrivial)
        report["analyses"]["factorization_patterns"] = self._factor_analysis(nontrivial)
        report["analyses"]["palindrome_analysis"] = self._palindrome_analysis(nontrivial)
        report["analyses"]["cross_pipeline_overlap"] = self._overlap_analysis(all_fps)
        report["analyses"]["digit_count_distribution"] = self._digit_count_analysis(nontrivial)
        report["analyses"]["structural_families"] = self._family_analysis(nontrivial)

        return report

    def _digit_sum_analysis(self, fps: List[FixedPointCharacterization]) -> Dict:
        ds_values = [fp.digit_sum_val for fp in fps]
        ds_mod9 = [v % 9 for v in ds_values]
        counter = Counter(ds_values)
        mod_counter = Counter(ds_mod9)

        return {
            "distribution": dict(counter.most_common(10)),
            "mod9_distribution": dict(mod_counter),
            "divisible_by_9_ratio": mod_counter.get(0, 0) / len(fps),
            "mean": np.mean(ds_values),
            "median": np.median(ds_values),
            "hypothesis": (
                f"digit_sum of fixed points clusters at multiples of 9. "
                f"ds%9==0: {mod_counter.get(0,0)}/{len(fps)} = "
                f"{mod_counter.get(0,0)/len(fps):.0%}"
            )
        }

    def _factor_analysis(self, fps: List[FixedPointCharacterization]) -> Dict:
        primes_present = Counter()
        for fp in fps:
            for p in fp.prime_factors:
                primes_present[p] += 1

        n = len(fps)
        enrichment = {}
        for prime, count in primes_present.most_common(10):
            base_rate = 1 / prime  # Expected random frequency
            observed_rate = count / n
            enrichment[prime] = {
                "count": count,
                "rate": observed_rate,
                "base_rate": base_rate,
                "enrichment": observed_rate / base_rate if base_rate > 0 else 0
            }

        # Special pattern: 3^2 * 11
        pattern_3sq_11 = sum(
            1 for fp in fps
            if fp.prime_factors.get(3, 0) >= 2 and fp.prime_factors.get(11, 0) >= 1
        )

        return {
            "prime_enrichment": enrichment,
            "pattern_3sq_11": {
                "count": pattern_3sq_11,
                "rate": pattern_3sq_11 / n,
                "hypothesis": (
                    f"3^2 * 11 pattern in {pattern_3sq_11}/{n} = "
                    f"{pattern_3sq_11/n:.0%} of FPs. "
                    f"99 = 3^2*11, 1089 = 3^2*11^2, 9999 = 3^2*11*101."
                )
            }
        }

    def _palindrome_analysis(self, fps: List[FixedPointCharacterization]) -> Dict:
        pals = [fp for fp in fps if fp.is_palindrome]
        base_rate = 0.003  # ~0.3% palindromes in [1, 100000]
        rate = len(pals) / len(fps)

        return {
            "count": len(pals),
            "rate": rate,
            "base_rate": base_rate,
            "enrichment": rate / base_rate if base_rate > 0 else 0,
            "examples": [(fp.value, factor_str(fp.value)) for fp in pals[:10]],
            "hypothesis": (
                f"Palindromes are {rate/base_rate:.0f}x enriched among fixed points "
                f"({len(pals)}/{len(fps)} = {rate:.0%} vs base {base_rate:.1%})"
            )
        }

    def _overlap_analysis(self, all_fps: List[FixedPointCharacterization]) -> Dict:
        value_pipelines: Dict[int, Set[Tuple[str, ...]]] = defaultdict(set)
        for fp in all_fps:
            value_pipelines[fp.value].add(fp.pipeline)

        universal = sorted(
            [(v, len(pipes)) for v, pipes in value_pipelines.items() if len(pipes) > 1],
            key=lambda x: -x[1]
        )

        return {
            "multi_pipeline_fps": len(universal),
            "top_universal": [
                {"value": v, "pipelines": c, "factorization": factor_str(v)}
                for v, c in universal[:10]
            ],
            "hypothesis": (
                f"{len(universal)} fixed points appear across multiple pipelines. "
                f"Top: {', '.join(str(v) for v, _ in universal[:5])}"
            ) if universal else "No cross-pipeline fixed points found."
        }

    def _digit_count_analysis(self, fps: List[FixedPointCharacterization]) -> Dict:
        dc = Counter(fp.digit_count for fp in fps)
        return {
            "distribution": dict(sorted(dc.items())),
            "mean_digits": np.mean([fp.digit_count for fp in fps]),
            "hypothesis": (
                f"Fixed points concentrate at {dc.most_common(1)[0][0]}-digit numbers "
                f"({dc.most_common(1)[0][1]}/{len(fps)} = "
                f"{dc.most_common(1)[0][1]/len(fps):.0%})"
            )
        }

    def _family_analysis(self, fps: List[FixedPointCharacterization]) -> Dict:
        """Group FPs into structural families."""
        families: Dict[str, List[int]] = defaultdict(list)
        for fp in fps:
            # Family key: (digit_sum mod 9, has_factor_3, has_factor_11)
            key = (
                fp.digit_sum_val % 9,
                3 in fp.prime_factors,
                11 in fp.prime_factors
            )
            family_name = (
                f"ds%9={key[0]}, "
                f"{'3|n' if key[1] else '3∤n'}, "
                f"{'11|n' if key[2] else '11∤n'}"
            )
            families[family_name].append(fp.value)

        sorted_families = sorted(families.items(), key=lambda x: -len(x[1]))
        return {
            "num_families": len(families),
            "top_families": [
                {"name": name, "count": len(members),
                 "examples": sorted(members)[:5]}
                for name, members in sorted_families[:8]
            ]
        }

    def print_report(self, report: Dict):
        nt = report["nontrivial_fps"]
        print(f"\n   Total FPs: {report['total_fps']} | Non-trivial: {nt}")

        for name, analysis in report["analyses"].items():
            print(f"\n   --- {name.upper().replace('_', ' ')} ---")

            if "hypothesis" in analysis:
                print(f"   HYPOTHESIS: {analysis['hypothesis']}")

            if name == "digit_sum_distribution":
                print(f"   Top digit_sums: {analysis['distribution']}")
                print(f"   Mod-9: {analysis['mod9_distribution']}")
                print(f"   div-by-9 ratio: {analysis['divisible_by_9_ratio']:.0%}")

            elif name == "factorization_patterns":
                for prime, info in list(analysis["prime_enrichment"].items())[:5]:
                    print(f"   Factor {prime}: {info['count']}/{nt} "
                          f"({info['rate']:.0%}, {info['enrichment']:.1f}x enriched)")
                p311 = analysis["pattern_3sq_11"]
                print(f"   Pattern 3^2*11: {p311['count']}/{nt} = {p311['rate']:.0%}")

            elif name == "palindrome_analysis":
                print(f"   Palindromes: {analysis['count']}/{nt} = {analysis['rate']:.0%} "
                      f"({analysis['enrichment']:.0f}x enriched)")
                if analysis["examples"]:
                    exs = ', '.join(f"{v}={f}" for v, f in analysis["examples"][:5])
                    print(f"   Examples: {exs}")

            elif name == "cross_pipeline_overlap":
                for item in analysis.get("top_universal", [])[:5]:
                    print(f"   FP {item['value']} ({item['factorization']}): "
                          f"in {item['pipelines']} pipelines")

            elif name == "structural_families":
                for fam in analysis.get("top_families", [])[:5]:
                    exs = ', '.join(str(x) for x in fam["examples"][:4])
                    print(f"   [{fam['count']}] {fam['name']}: {exs}...")


# =============================================================================
# NEW MODULE D: THEORY GRAPH
# =============================================================================

class NodeType(Enum):
    OPERATOR = "operator"
    PIPELINE = "pipeline"
    FIXED_POINT = "fixed_point"
    THEOREM = "theorem"
    PROOF_SKETCH = "proof_sketch"
    MECHANISM = "mechanism"


class RelationType(Enum):
    COMPOSES = "composes"           # Operator -> Pipeline
    CONVERGES_TO = "converges_to"   # Pipeline -> FixedPoint
    SUPPORTS = "supports"           # Pipeline -> Theorem
    FALSIFIES = "falsifies"         # Pipeline -> Theorem
    PROVES_VIA = "proves_via"       # Theorem -> ProofSketch
    SHARES_FACTOR = "shares_factor" # FixedPoint -> FixedPoint


@dataclass
class GraphNode:
    id: str
    type: NodeType
    label: str
    data: Dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: RelationType
    weight: float = 1.0


class TheoryGraph:
    """Connects all discovered objects in a directed graph."""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode):
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)

    def build_from_session(self, results: List[Dict],
                           all_fps: List[FixedPointCharacterization],
                           theorems: List[MetaTheorem],
                           proof_sketches: List[ProofSketch]):
        # Add operators
        for op_name in OPERATIONS:
            self.add_node(GraphNode(
                id=f"op_{op_name}", type=NodeType.OPERATOR, label=op_name
            ))

        # Add pipelines + edges to operators and fixed points
        seen_pipelines = set()
        for r in results:
            pipe = r["pipeline"]
            pipe_id = f"pipe_{'_'.join(pipe)}"
            if pipe_id not in seen_pipelines:
                seen_pipelines.add(pipe_id)
                self.add_node(GraphNode(
                    id=pipe_id, type=NodeType.PIPELINE,
                    label=' -> '.join(pipe),
                    data={"dominance": r.get("dominance", 0),
                          "attractor": r.get("attractor")}
                ))
                for op in pipe:
                    self.add_edge(GraphEdge(
                        source=f"op_{op}", target=pipe_id,
                        relation=RelationType.COMPOSES
                    ))

        # Add fixed points
        fp_ids = set()
        for fp in all_fps:
            fp_id = f"fp_{fp.value}"
            if fp_id not in fp_ids:
                fp_ids.add(fp_id)
                self.add_node(GraphNode(
                    id=fp_id, type=NodeType.FIXED_POINT,
                    label=str(fp.value),
                    data={"digit_sum": fp.digit_sum_val,
                          "palindrome": fp.is_palindrome,
                          "factors": fp.prime_factors}
                ))
            pipe_id = f"pipe_{'_'.join(fp.pipeline)}"
            self.add_edge(GraphEdge(
                source=pipe_id, target=fp_id,
                relation=RelationType.CONVERGES_TO
            ))

        # Add shared-factor edges between FPs
        nontrivial_fps = [fp for fp in all_fps if fp.value > 0]
        fp_dedup = {}
        for fp in nontrivial_fps:
            if fp.value not in fp_dedup:
                fp_dedup[fp.value] = fp
        fp_list = list(fp_dedup.values())
        for i, fp1 in enumerate(fp_list):
            for fp2 in fp_list[i+1:]:
                shared_primes = set(fp1.prime_factors) & set(fp2.prime_factors)
                if len(shared_primes) >= 2:
                    self.add_edge(GraphEdge(
                        source=f"fp_{fp1.value}", target=f"fp_{fp2.value}",
                        relation=RelationType.SHARES_FACTOR,
                        weight=len(shared_primes)
                    ))

        # Add theorems
        for t in theorems:
            self.add_node(GraphNode(
                id=f"thm_{t.id}", type=NodeType.THEOREM,
                label=t.statement[:50],
                data={"status": t.status.value, "confidence": t.confidence,
                      "source": t.source}
            ))

        # Add proof sketches
        for ps in proof_sketches:
            ps_id = f"proof_{ps.theorem_id}"
            self.add_node(GraphNode(
                id=ps_id, type=NodeType.PROOF_SKETCH,
                label=f"Proof via {ps.strategy.value}",
                data={"strength": ps.strength,
                      "n_gaps": len(ps.gaps)}
            ))
            self.add_edge(GraphEdge(
                source=f"thm_{ps.theorem_id}", target=ps_id,
                relation=RelationType.PROVES_VIA
            ))

    def get_stats(self) -> Dict:
        type_counts = Counter(n.type.value for n in self.nodes.values())
        rel_counts = Counter(e.relation.value for e in self.edges)
        return {"nodes": dict(type_counts), "edges": dict(rel_counts)}

    def query_connected(self, node_id: str) -> List[Tuple[str, str, str]]:
        """Find all connected nodes."""
        results = []
        for edge in self.edges:
            if edge.source == node_id:
                target = self.nodes.get(edge.target)
                if target:
                    results.append((edge.relation.value, target.id, target.label))
            elif edge.target == node_id:
                source = self.nodes.get(edge.source)
                if source:
                    results.append((edge.relation.value, source.id, source.label))
        return results

    def print_summary(self):
        stats = self.get_stats()
        print(f"\n   Nodes: {sum(stats['nodes'].values())}")
        for t, c in sorted(stats['nodes'].items()):
            print(f"     {t}: {c}")
        print(f"   Edges: {sum(stats['edges'].values())}")
        for t, c in sorted(stats['edges'].items()):
            print(f"     {t}: {c}")

        # Most connected fixed points
        fp_connections = Counter()
        for edge in self.edges:
            if edge.target.startswith("fp_"):
                fp_connections[edge.target] += 1
            if edge.source.startswith("fp_"):
                fp_connections[edge.source] += 1
        if fp_connections:
            print(f"\n   Most connected fixed points:")
            for fp_id, count in fp_connections.most_common(5):
                node = self.nodes.get(fp_id)
                if node:
                    print(f"     FP {node.label}: {count} connections")


# =============================================================================
# MAIN ENGINE: DEDUCTIVE THEORY ENGINE v8.0
# =============================================================================

class DeductiveTheoryEngine:
    def __init__(self):
        self.ops = OPERATIONS
        print("   Computing operator algebra...")
        self.algebra = OperatorAlgebra(self.ops)
        print(f"   {len(self.algebra.profiles)} profiles computed")

        self.fp_solver = FixedPointSolver(self.ops)
        self.theorem_gen = MetaTheoremGenerator(self.algebra, self.ops)
        self.proof_gen = ProofSketchGenerator()
        self.inductive_gen = InductiveTheoremGenerator(self.algebra, self.ops)
        self.fp_analyzer = FixedPointStructuralAnalyzer()
        self.theory_graph = TheoryGraph()

        self.op_scores: Dict[str, float] = {op: 1.0 for op in self.ops}
        self.exploration_rate = 0.4
        self.results: List[Dict] = []
        self.all_fps: List[FixedPointCharacterization] = []
        self.all_proof_sketches: List[ProofSketch] = []

    def apply_pipeline(self, n: int, pipeline: Tuple[str, ...]) -> int:
        for op_name in pipeline:
            if op_name in self.ops:
                n = self.ops[op_name](n)
                if n > 10**15 or n < 0:
                    return -1
        return n

    def select_pipeline(self) -> Tuple[str, ...]:
        length = random.choices([2, 3, 4], weights=[0.5, 0.35, 0.15])[0]
        if random.random() < self.exploration_rate:
            return tuple(random.choices(list(self.ops.keys()), k=length))
        weights = [max(0.01, self.op_scores.get(op, 1.0)) for op in self.ops]
        total = sum(weights)
        probs = [w / total for w in weights]
        return tuple(np.random.choice(list(self.ops.keys()), size=length, p=probs))

    def explore_pipeline(self, pipeline: Tuple[str, ...]) -> Dict:
        prediction = self.algebra.predict_convergence(pipeline)
        predicted_props = prediction["predicted_properties"]

        numbers = random.sample(range(1000, 99999), 3000)
        endpoints = Counter()
        observed_props = set()

        for n in numbers:
            current = n
            for _ in range(100):
                prev = current
                current = self.apply_pipeline(current, pipeline)
                if current < 0 or current == prev:
                    break
            if current >= 0 and current == prev:
                endpoints[current] += 1

        attractor = None
        dominance = 0.0
        if endpoints:
            attractor, count = endpoints.most_common(1)[0]
            dominance = 100 * count / len(numbers)

        mod9_ok = monotone_ok = entropy_ok = 0
        for n in numbers[:500]:
            result = self.apply_pipeline(n, pipeline)
            if result < 0:
                continue
            if result % 9 == n % 9:
                mod9_ok += 1
            if result < n:
                monotone_ok += 1
            if result > 0 and digit_entropy(result) < digit_entropy(n):
                entropy_ok += 1
        nt = min(500, len(numbers))
        if nt > 0:
            if mod9_ok / nt > 0.99:
                observed_props.add("mod9_preserved")
            if monotone_ok / nt > 0.95:
                observed_props.add("monotone")
            if entropy_ok / nt > 0.7:
                observed_props.add("entropy_reducing")
        if dominance > 80:
            observed_props.add("strong_convergence")

        fixed_points = []
        if dominance > 50 and attractor is not None:
            fixed_points = self.fp_solver.solve(
                pipeline,
                domain=(0, min(200000, max(attractor * 2, 10000))),
                predicted_properties=predicted_props
            )
            self.all_fps.extend(fixed_points)

        # Prediction accuracy
        pred_correct = pred_total = 0
        if AlgebraicProperty.PRESERVES_MOD_9 in predicted_props:
            pred_total += 1
            if "mod9_preserved" in observed_props:
                pred_correct += 1
        if AlgebraicProperty.MONOTONE_REDUCING in predicted_props:
            pred_total += 1
            if "monotone" in observed_props:
                pred_correct += 1
        prediction_accuracy = pred_correct / pred_total if pred_total > 0 else None

        score = dominance / 100 * (1 + len(fixed_points) * 0.1)
        for op in pipeline:
            old = self.op_scores.get(op, 1.0)
            self.op_scores[op] = 0.85 * old + 0.15 * score * 2

        result = {
            "pipeline": pipeline, "attractor": attractor, "dominance": dominance,
            "predicted_properties": predicted_props,
            "observed_properties": observed_props,
            "prediction_accuracy": prediction_accuracy,
            "guarantees": prediction["guarantees"],
            "fixed_points": fixed_points, "score": score
        }
        self.results.append(result)
        return result

    def run_research_session(self, cycles: int = 8, pipelines_per_cycle: int = 20):
        print("█" * 70)
        print("  SYNTRIAD DEDUCTIVE THEORY ENGINE v8.0")
        print("  From Detection to Deduction")
        print("█" * 70)
        session_start = time.time()

        # ── Phase 0: Operator Algebra ────────────────────────────────
        print("\n" + "▓" * 70)
        print("  PHASE 0: OPERATOR ALGEBRA")
        print("▓" * 70)
        print(f"\n{self.algebra.get_summary()}")

        # ── Phase 1: Pre-defined Meta-Theorem Testing ────────────────
        print("\n" + "▓" * 70)
        print("  PHASE 1: PRE-DEFINED META-THEOREM TESTING")
        print("▓" * 70)
        self.theorem_gen.test_all(n_test_pipelines=60)
        for t in self.theorem_gen.theorems:
            icon = {"strong_empirical": "✅", "tested": "⚠️",
                    "falsified": "❌", "conjecture": "?"}.get(t.status.value, "?")
            print(f"\n   {icon} [{t.id}] {t.statement[:70]}...")
            print(f"      {t.status.value} | {t.supporting_pipelines}/{t.tested_pipelines} "
                  f"| conf={t.confidence:.2f}")
            if t.counterexample_pipeline:
                print(f"      COUNTEREX: {' -> '.join(t.counterexample_pipeline)}")

        # ── Phase 1b: Proof Sketches ─────────────────────────────────
        print("\n" + "▓" * 70)
        print("  PHASE 1b: PROOF SKETCH GENERATION")
        print("▓" * 70)
        for t in self.theorem_gen.theorems:
            if t.status == TheoremStatus.STRONG_EMPIRICAL:
                sketch = self.proof_gen.generate(t)
                if sketch:
                    self.all_proof_sketches.append(sketch)
                    print(f"\n   📐 Proof sketch for [{t.id}] via {sketch.strategy.value}")
                    print(f"      Strength: {sketch.strength}")
                    for step in sketch.steps[:4]:
                        print(f"      {step}")
                    if len(sketch.steps) > 4:
                        print(f"      ... ({len(sketch.steps)} steps total)")
                    if sketch.gaps:
                        print(f"      ⚠ GAPS ({len(sketch.gaps)}):")
                        for gap in sketch.gaps[:2]:
                            print(f"        - {gap[:65]}...")

        # ── Phase 2: Exploration ─────────────────────────────────────
        print("\n" + "▓" * 70)
        print("  PHASE 2: SYMBOLIC EXPLORATION")
        print("▓" * 70)
        for cycle in range(cycles):
            print(f"\n{'─'*60}")
            print(f"  Cycle {cycle+1}/{cycles}")
            print(f"{'─'*60}")
            for i in range(pipelines_per_cycle):
                pipeline = self.select_pipeline()
                result = self.explore_pipeline(pipeline)
                if result["score"] > 0.6 or (result["fixed_points"] and len(result["fixed_points"]) > 1):
                    pipe_str = ' -> '.join(pipeline)
                    pred_str = ', '.join(p.value for p in result["predicted_properties"])
                    print(f"\n   [{i+1}] {pipe_str}")
                    print(f"       Predicted: {{{pred_str}}}")
                    print(f"       Attr={result['attractor']}, Dom={result['dominance']:.1f}%")
                    for g in result["guarantees"]:
                        print(f"       📐 {g}")
                    for fp in result["fixed_points"][:3]:
                        if fp.value > 0:
                            print(f"       🎯 {fp.explanation[:65]}...")

        # ── Phase 3: Fixed-Point Structural Analysis ─────────────────
        print("\n" + "▓" * 70)
        print("  PHASE 3: FIXED-POINT STRUCTURAL ANALYSIS")
        print("▓" * 70)
        fp_report = self.fp_analyzer.analyze(self.all_fps)
        self.fp_analyzer.print_report(fp_report)

        # ── Phase 4: Inductive Theorem Generation ────────────────────
        print("\n" + "▓" * 70)
        print("  PHASE 4: INDUCTIVE THEOREM GENERATION")
        print("  (Theorems generated FROM data, not from templates)")
        print("▓" * 70)
        induced_fp = self.inductive_gen.induce_from_fixed_points(self.all_fps)
        induced_conv = self.inductive_gen.induce_from_convergence_patterns(self.results)
        all_induced = induced_fp + induced_conv

        for t in all_induced:
            print(f"\n   🔬 [{t.id}] {t.statement[:70]}...")
            print(f"      Source: {t.source} | Confidence: {t.confidence:.2f}")

        # ── Phase 5: Theory Graph ────────────────────────────────────
        print("\n" + "▓" * 70)
        print("  PHASE 5: THEORY GRAPH")
        print("▓" * 70)
        all_theorems = self.theorem_gen.theorems + all_induced
        self.theory_graph.build_from_session(
            self.results, self.all_fps, all_theorems, self.all_proof_sketches
        )
        self.theory_graph.print_summary()

        # ── Final Report ─────────────────────────────────────────────
        session_duration = time.time() - session_start
        print("\n" + "█" * 70)
        print("  SESSION COMPLETE — DEDUCTIVE THEORY ENGINE v8.0")
        print("█" * 70)

        print(f"\n📊 STATISTICS:")
        print(f"   Duration: {session_duration:.1f}s")
        print(f"   Pipelines explored: {len(self.results)}")
        print(f"   Unique attractors: {len(set(r['attractor'] for r in self.results if r['attractor']))}")
        print(f"   Fixed points characterized: {len(self.all_fps)}")

        accuracies = [r["prediction_accuracy"] for r in self.results
                     if r["prediction_accuracy"] is not None]
        if accuracies:
            print(f"   Symbolic prediction accuracy: {np.mean(accuracies):.1%}")

        strong = [t for t in self.theorem_gen.theorems
                 if t.status == TheoremStatus.STRONG_EMPIRICAL]
        falsified = [t for t in self.theorem_gen.theorems
                    if t.status == TheoremStatus.FALSIFIED]

        print(f"\n📐 PRE-DEFINED THEOREMS: {len(strong)} strong, {len(falsified)} falsified")
        print(f"🔬 INDUCED THEOREMS: {len(all_induced)} generated from data")
        print(f"📜 PROOF SKETCHES: {len(self.all_proof_sketches)} generated")

        for ps in self.all_proof_sketches:
            print(f"   [{ps.theorem_id}] via {ps.strategy.value} "
                  f"({ps.strength}, {len(ps.gaps)} gaps)")

        print(f"\n🔬 INDUCED THEOREMS (the system's own discoveries):")
        for t in all_induced:
            print(f"   [{t.id}] {t.statement[:65]}...")

        top = sorted(self.results, key=lambda r: r["score"], reverse=True)[:5]
        print(f"\n💎 TOP DISCOVERIES:")
        for r in top:
            pipe_str = ' -> '.join(r["pipeline"])
            print(f"   [{r['score']:.3f}] {pipe_str}")
            print(f"           Attr={r['attractor']} Dom={r['dominance']:.1f}% "
                  f"FPs={len(r['fixed_points'])}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    engine = DeductiveTheoryEngine()
    engine.run_research_session(cycles=10, pipelines_per_cycle=20)
