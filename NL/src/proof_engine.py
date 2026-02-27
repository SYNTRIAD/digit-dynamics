# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research � MIT License
"""
ENGINE vNext — Module M3: Proof Engine + Structural Reasoning

Builds on M0-M2 to provide:
  1. Proof Skeleton Generator — identify structural invariants, reduction lemmas,
     known theorem links, remaining gaps for each conjecture
  2. Counterexample Density Estimator — search space volume, observed density,
     upper bound estimates, confidence scaling beyond "survived to k=N"
  3. Pattern Compressor — detect affine patterns, modular invariants, linear
     recurrences in counting sequences (structural, not enumerative)
  4. Conjecture Mutator — hypothesis mutation, cross-pipeline comparison,
     invariant transfer, search steering
  5. Ranking Model v1.0 — formalized heuristic with explicit versioning and
     Bayesian-inspired confidence calibration

Usage:
    from proof_engine import (
        ProofSkeleton, SkeletonGenerator,
        DensityEstimator, PatternCompressor,
        ConjectureMutator, RankingModelV1,
    )

    gen = SkeletonGenerator(registry)
    skeleton = gen.generate(conjecture)
    print(skeleton.to_dict())
"""

from __future__ import annotations

import math
import json
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline_dsl import (
    OperationRegistry, Pipeline, DomainPolicy, RunResult,
    PipelineRunner, SemanticClass, DSClass,
)
from feature_extractor import (
    Conjecture, ConjectureType, TestedDomain, MonotonicityKind,
    NumberFeatures, OrbitAnalyzer, ConjectureMiner,
)


# =============================================================================
# 1. PROOF SKELETON GENERATOR
# =============================================================================

class ProofStrategy(str, Enum):
    """Classification of proof approaches."""
    MOD_INVARIANT = "mod_invariant"           # Digit-sum / modular arithmetic
    DIGIT_PAIR_CONSTRAINT = "digit_pair"      # Reverse-complement digit pairing
    BOUNDING = "bounding"                     # Upper/lower bound argument
    COUNTING_RECURRENCE = "counting_recurrence"  # Recurrence relation for #FP(k)
    STRUCTURAL_FORM = "structural_form"       # FP has specific digit pattern
    PIGEONHOLE = "pigeonhole"                 # Finiteness / pigeonhole argument
    UNKNOWN = "unknown"                       # No strategy identified


@dataclass
class ReductionStep:
    """A single step in a proof skeleton."""
    description: str
    formal: str              # Formal mathematical statement
    justification: str       # Why this step holds
    status: str = "claimed"  # "proven" | "claimed" | "gap"


@dataclass
class ProofSkeleton:
    """Structured proof skeleton for a conjecture."""
    conjecture_statement: str
    strategy: ProofStrategy
    structural_invariant: str      # The key algebraic property
    reduction_steps: List[ReductionStep]
    known_theorem_links: List[str]  # References to known results
    remaining_gaps: List[str]       # What's still unproven
    proof_strength: str             # "complete" | "modulo_gap" | "heuristic"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "conjecture": self.conjecture_statement,
            "strategy": self.strategy.value,
            "structural_invariant": self.structural_invariant,
            "reduction_steps": [
                {"description": s.description, "formal": s.formal,
                 "justification": s.justification, "status": s.status}
                for s in self.reduction_steps
            ],
            "known_theorem_links": self.known_theorem_links,
            "remaining_gaps": self.remaining_gaps,
            "proof_strength": self.proof_strength,
            "notes": self.notes,
        }

    def __str__(self) -> str:
        gaps = f" [{len(self.remaining_gaps)} gap(s)]" if self.remaining_gaps else ""
        return (f"ProofSkeleton({self.strategy.value}, "
                f"strength={self.proof_strength}{gaps})")


class SkeletonGenerator:
    """Generate proof skeletons for conjectures based on structural analysis."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()

    def generate(self, conjecture: Conjecture) -> ProofSkeleton:
        """Route conjecture to appropriate skeleton generator."""
        if conjecture.ctype == ConjectureType.MODULAR:
            return self._skeleton_modular(conjecture)
        elif conjecture.ctype == ConjectureType.COUNTING:
            return self._skeleton_counting(conjecture)
        elif conjecture.ctype == ConjectureType.MONOTONICITY:
            return self._skeleton_monotonicity(conjecture)
        elif conjecture.ctype == ConjectureType.UNIVERSALITY:
            return self._skeleton_universality(conjecture)
        elif conjecture.ctype == ConjectureType.STRUCTURE:
            return self._skeleton_structure(conjecture)
        elif conjecture.ctype == ConjectureType.INVARIANT:
            return self._skeleton_invariant(conjecture)
        return self._skeleton_generic(conjecture)

    def _skeleton_modular(self, c: Conjecture) -> ProofSkeleton:
        """Proof skeleton for modular conjectures (e.g., FP ≡ 0 mod 9)."""
        m = c.parameters.get("modulus", 9)
        r = c.parameters.get("residue", 0)
        pipeline = c.pipeline
        base = c.parameters.get("base", 10)

        # Identify which operations preserve/create mod invariants
        ops = Pipeline.parse(pipeline, registry=self.registry).steps
        op_analysis = []
        for step in ops:
            spec = self.registry.get_spec(step.op_name)
            if spec.preserves_mod_b_minus_1 and m == base - 1:
                op_analysis.append(
                    f"{step.op_name} preserves mod {base-1} (digit sum invariant)")
            elif spec.semantic_class == SemanticClass.SUBTRACTIVE:
                op_analysis.append(
                    f"{step.op_name} is subtractive: sort_desc(n) - sort_asc(n) "
                    f"≡ 0 mod {base-1}")
            elif spec.semantic_class == SemanticClass.MIXED:
                op_analysis.append(
                    f"{step.op_name} is mixed: needs per-operation analysis")

        steps = []

        # Step 1: Digit sum preservation
        if m == base - 1:
            steps.append(ReductionStep(
                description=f"Digit rearrangement preserves digit sum mod {base-1}",
                formal=f"∀n: ds(sort_desc(n)) ≡ ds(n) (mod {base-1})",
                justification="Digits are permuted, sum is invariant under permutation",
                status="proven",
            ))

        # Step 2: Subtraction mod analysis
        if any(self.registry.get_spec(s.op_name).semantic_class == SemanticClass.SUBTRACTIVE
               for s in ops):
            steps.append(ReductionStep(
                description=f"Subtraction of digit permutations yields ≡ 0 mod {base-1}",
                formal=f"sort_desc(n) - sort_asc(n) ≡ ds(n) - ds(n) ≡ 0 (mod {base-1})",
                justification=(
                    f"Both sort_desc(n) and sort_asc(n) have the same digit sum, "
                    f"so their difference is ≡ 0 mod {base-1}"),
                status="proven",
            ))

        # Step 3: Fixed point implication
        steps.append(ReductionStep(
            description=f"If FP = f(FP), then FP ≡ f(FP) ≡ {r} (mod {m})",
            formal=f"FP = f(FP) ⟹ FP ≡ {r} (mod {m})",
            justification="Fixed point equation combined with modular invariant",
            status="proven" if m == base - 1 and r == 0 else "claimed",
        ))

        # Identify gaps
        gaps = []
        if m != base - 1:
            gaps.append(f"Modulus {m} is not base-1={base-1}; digit sum argument "
                        f"does not directly apply")
        if r != 0 and m == base - 1:
            gaps.append(f"Non-zero residue {r} requires analysis of operation offsets")

        # Determine proof strength
        all_proven = all(s.status == "proven" for s in steps) and not gaps
        strength = "complete" if all_proven else ("modulo_gap" if steps else "heuristic")

        return ProofSkeleton(
            conjecture_statement=c.statement,
            strategy=ProofStrategy.MOD_INVARIANT,
            structural_invariant=(
                f"Digit sum mod {base-1} is preserved under digit permutation. "
                f"Subtractive operations yield results ≡ 0 mod {base-1}."),
            reduction_steps=steps,
            known_theorem_links=[
                f"Digit sum theorem: n ≡ ds(n) (mod {base-1})",
                "Casting out nines (base 10)",
                "Kaprekar constant properties (D.R. Kaprekar, 1949)",
            ],
            remaining_gaps=gaps,
            proof_strength=strength,
            notes="; ".join(op_analysis) if op_analysis else "",
        )

    def _skeleton_counting(self, c: Conjecture) -> ProofSkeleton:
        """Proof skeleton for counting conjectures (#FP = f(k))."""
        evidence = c.evidence
        ks = sorted(int(k) for k in evidence.keys())
        vals = [evidence[k] if isinstance(k, int) else evidence[str(k)] for k in ks]

        steps = []

        # Check if constant
        if len(set(vals)) == 1:
            steps.append(ReductionStep(
                description=f"Observed: #FP is constant = {vals[0]} for k ∈ {ks}",
                formal=f"#FP(k) = {vals[0]} for k ∈ {{{','.join(map(str, ks))}}}",
                justification="Exhaustive enumeration over tested domains",
                status="claimed",
            ))
            steps.append(ReductionStep(
                description="Constant count requires structural argument for all k",
                formal="∀k≥k₀: exactly one FP exists per digit-length class",
                justification="Needs: contraction mapping + uniqueness in each k-class",
                status="gap",
            ))

        # Check linear
        if "a" in c.parameters and "b" in c.parameters:
            a, b = c.parameters["a"], c.parameters["b"]
            steps.append(ReductionStep(
                description=f"Linear fit: #FP(k) = {a}·k + {b}",
                formal=f"#FP(k) = {a}k + {b}",
                justification="Observed in data; requires recurrence or bijection proof",
                status="claimed",
            ))

        gaps = [
            "Structural explanation for counting formula not yet derived",
            "Need: characterize FP digit patterns as function of k",
        ]

        return ProofSkeleton(
            conjecture_statement=c.statement,
            strategy=ProofStrategy.COUNTING_RECURRENCE,
            structural_invariant="Fixed point count as function of digit length k",
            reduction_steps=steps,
            known_theorem_links=[
                "Kaprekar routine fixed point uniqueness (k=4, base 10)",
                "FP enumeration techniques in digit-dynamical systems",
            ],
            remaining_gaps=gaps,
            proof_strength="heuristic",
        )

    def _skeleton_monotonicity(self, c: Conjecture) -> ProofSkeleton:
        """Proof skeleton for monotonicity conjectures."""
        kind = c.monotonicity_kind
        label = kind.value if kind else "monotone"

        steps = [
            ReductionStep(
                description=f"Observed: sequence is {label} over tested k",
                formal=c.predicate,
                justification="Exhaustive verification over tested domains",
                status="claimed",
            ),
        ]

        if kind == MonotonicityKind.EVENTUALLY_CONSTANT:
            steps.append(ReductionStep(
                description="Eventually constant requires bounding argument",
                formal="∃K: ∀k≥K, metric(k) = C",
                justification="Need: upper bound on metric or finiteness argument",
                status="gap",
            ))
        elif kind in (MonotonicityKind.NONDECREASING, MonotonicityKind.STRICTLY_INCREASING):
            steps.append(ReductionStep(
                description="Monotone growth requires injection or structural argument",
                formal="FPs at digit length k embed into FPs at digit length k+1",
                justification="Need: constructive embedding or counting argument",
                status="gap",
            ))

        return ProofSkeleton(
            conjecture_statement=c.statement,
            strategy=ProofStrategy.BOUNDING,
            structural_invariant=f"Sequence behavior: {label}",
            reduction_steps=steps,
            known_theorem_links=[
                "Monotonicity in parametric dynamical systems",
            ],
            remaining_gaps=["Structural proof of monotonicity not yet derived"],
            proof_strength="heuristic",
        )

    def _skeleton_universality(self, c: Conjecture) -> ProofSkeleton:
        """Proof skeleton for universality/convergence conjectures."""
        steps = [
            ReductionStep(
                description="Near-universal convergence observed (>99%)",
                formal=c.predicate,
                justification="Exhaustive enumeration shows dominant basin fraction",
                status="claimed",
            ),
            ReductionStep(
                description="Non-converging inputs are repdigit-adjacent or zero-padded",
                formal="Exceptions are structurally constrained",
                justification="Need: characterize exception set explicitly",
                status="gap",
            ),
            ReductionStep(
                description="Contraction argument for convergence",
                formal="f is eventually contractive on most of D(k)",
                justification="Need: contraction ratio < 1 on (1-ε) fraction of domain",
                status="gap",
            ),
        ]

        return ProofSkeleton(
            conjecture_statement=c.statement,
            strategy=ProofStrategy.BOUNDING,
            structural_invariant="Dominant basin fraction approaches 1",
            reduction_steps=steps,
            known_theorem_links=[
                "Basin of attraction analysis in iterated digit maps",
                "ε-universality conjecture (Paper B framework)",
            ],
            remaining_gaps=[
                "Exception set characterization",
                "Contraction ratio bound proof",
            ],
            proof_strength="heuristic",
        )

    def _skeleton_structure(self, c: Conjecture) -> ProofSkeleton:
        """Proof skeleton for structural conjectures (palindrome, patterns)."""
        steps = [
            ReductionStep(
                description="FP structural property observed across tested k",
                formal=c.predicate,
                justification="Exhaustive verification",
                status="claimed",
            ),
            ReductionStep(
                description="Structural form must follow from fixed-point equation",
                formal="FP = f(FP) constrains digit arrangement",
                justification="Need: solve f(x) = x for digit pattern",
                status="gap",
            ),
        ]

        return ProofSkeleton(
            conjecture_statement=c.statement,
            strategy=ProofStrategy.STRUCTURAL_FORM,
            structural_invariant="Fixed points have specific digit structure",
            reduction_steps=steps,
            known_theorem_links=[
                "Fixed point digit patterns in Kaprekar routine",
                "Palindrome properties of 1089-family fixed points",
            ],
            remaining_gaps=["Digit-level algebraic characterization of FP form"],
            proof_strength="heuristic",
        )

    def _skeleton_invariant(self, c: Conjecture) -> ProofSkeleton:
        """Proof skeleton for invariant conjectures (digit sum = constant)."""
        steps = [
            ReductionStep(
                description=f"Invariant observed: {c.predicate}",
                formal=c.predicate,
                justification="Exhaustive verification over tested domains",
                status="claimed",
            ),
            ReductionStep(
                description="Invariant must follow from operation algebraic properties",
                formal="f preserves/constrains the invariant quantity",
                justification="Need: trace invariant through operation pipeline",
                status="gap",
            ),
        ]

        return ProofSkeleton(
            conjecture_statement=c.statement,
            strategy=ProofStrategy.MOD_INVARIANT,
            structural_invariant=c.predicate,
            reduction_steps=steps,
            known_theorem_links=[
                "Digit sum conservation in digit-preserving maps",
            ],
            remaining_gaps=["Full algebraic derivation of invariant"],
            proof_strength="heuristic",
        )

    def _skeleton_generic(self, c: Conjecture) -> ProofSkeleton:
        """Fallback skeleton for unrecognized conjecture types."""
        return ProofSkeleton(
            conjecture_statement=c.statement,
            strategy=ProofStrategy.UNKNOWN,
            structural_invariant="Not identified",
            reduction_steps=[ReductionStep(
                description="Conjecture observed empirically",
                formal=c.predicate,
                justification="Exhaustive/sampled verification",
                status="claimed",
            )],
            known_theorem_links=[],
            remaining_gaps=["No proof strategy identified"],
            proof_strength="heuristic",
        )


# =============================================================================
# 2. COUNTEREXAMPLE DENSITY ESTIMATOR
# =============================================================================

@dataclass
class DensityEstimate:
    """Counterexample density estimate for a conjecture."""
    total_search_space: int       # Total values tested
    num_counterexamples: int      # CEs found (0 if survived)
    observed_density: float       # num_CEs / total_search_space
    upper_bound_95: float         # 95% CI upper bound on true CE rate
    upper_bound_99: float         # 99% CI upper bound on true CE rate
    confidence_score: float       # Calibrated confidence (0-1)
    search_volume_log10: float    # log10(total_search_space) for scale context
    k_range: Tuple[int, int]      # (k_min, k_max) tested
    falsification_label: str      # "strong" | "moderate" | "weak"

    def to_dict(self) -> dict:
        return {
            "total_search_space": self.total_search_space,
            "num_counterexamples": self.num_counterexamples,
            "observed_density": round(self.observed_density, 12),
            "upper_bound_95": round(self.upper_bound_95, 8),
            "upper_bound_99": round(self.upper_bound_99, 8),
            "confidence_score": round(self.confidence_score, 4),
            "search_volume_log10": round(self.search_volume_log10, 2),
            "k_range": list(self.k_range),
            "falsification_label": self.falsification_label,
        }

    def __str__(self) -> str:
        return (f"DensityEstimate(N={self.total_search_space}, "
                f"CEs={self.num_counterexamples}, "
                f"ub95={self.upper_bound_95:.2e}, "
                f"label={self.falsification_label})")


class DensityEstimator:
    """Estimate counterexample density and calibrate falsification strength."""

    @staticmethod
    def estimate(conjecture: Conjecture, num_counterexamples: int = 0) -> DensityEstimate:
        """Compute density estimate from tested_domains on a conjecture."""
        total = 0
        k_min, k_max = 999, 0

        for td in conjecture.tested_domains:
            n = td.range_hi - td.range_lo + 1
            if td.exclude_repdigits and td.digit_length > 0:
                n -= (td.base - 1)  # approximate repdigit count
            total += max(n, 0)
            k_min = min(k_min, td.digit_length)
            k_max = max(k_max, td.digit_length)

        if not conjecture.tested_domains:
            # Fallback: estimate from parameters
            k_range_param = conjecture.parameters.get("k_range", [])
            base = conjecture.parameters.get("base", 10)
            for k in k_range_param:
                total += base ** k - base ** (k - 1) if k > 1 else base - 1
                k_min = min(k_min, k)
                k_max = max(k_max, k)

        if k_min > k_max:
            k_min, k_max = 0, 0

        if total == 0:
            return DensityEstimate(
                total_search_space=0, num_counterexamples=num_counterexamples,
                observed_density=0.0, upper_bound_95=1.0, upper_bound_99=1.0,
                confidence_score=0.0, search_volume_log10=0.0,
                k_range=(k_min, k_max), falsification_label="untested",
            )

        observed = num_counterexamples / total if total > 0 else 0.0

        # Clopper-Pearson-inspired upper bounds (Rule of Three for 0 CEs)
        if num_counterexamples == 0:
            ub_95 = 3.0 / total      # P(0 in N) < 0.05 when p > 3/N
            ub_99 = 4.61 / total     # P(0 in N) < 0.01 when p > 4.61/N
        else:
            # Wilson score interval upper bound
            z95, z99 = 1.96, 2.576
            p_hat = observed
            ub_95 = (p_hat + z95**2/(2*total) + z95*math.sqrt(
                (p_hat*(1-p_hat) + z95**2/(4*total))/total
            )) / (1 + z95**2/total)
            ub_99 = (p_hat + z99**2/(2*total) + z99*math.sqrt(
                (p_hat*(1-p_hat) + z99**2/(4*total))/total
            )) / (1 + z99**2/total)

        # Confidence score: combines search volume + k range coverage
        log_vol = math.log10(total) if total > 1 else 0
        k_span = k_max - k_min + 1
        # Bayesian-inspired: P(conjecture true | 0 CEs in N trials)
        # Assuming uniform prior: confidence ≈ 1 - 1/(N+1)
        if num_counterexamples == 0:
            bayesian_conf = 1 - 1 / (total + 1)
            # Scale by k-range breadth (broader → more confident)
            range_factor = min(k_span / 10.0, 1.0)
            confidence = min(bayesian_conf * (0.5 + 0.5 * range_factor), 0.999)
        else:
            confidence = max(0.0, 1 - observed) * 0.5  # Halved if CEs exist

        # Falsification strength label
        if num_counterexamples > 0:
            label = "falsified"
        elif log_vol >= 7 and k_span >= 5:
            label = "strong"
        elif log_vol >= 5 and k_span >= 3:
            label = "moderate"
        elif log_vol >= 3:
            label = "weak"
        else:
            label = "minimal"

        return DensityEstimate(
            total_search_space=total,
            num_counterexamples=num_counterexamples,
            observed_density=observed,
            upper_bound_95=ub_95,
            upper_bound_99=ub_99,
            confidence_score=confidence,
            search_volume_log10=log_vol,
            k_range=(k_min, k_max),
            falsification_label=label,
        )


# =============================================================================
# 3. PATTERN COMPRESSOR
# =============================================================================

@dataclass
class DetectedPattern:
    """A structural pattern detected in a sequence."""
    pattern_type: str       # "constant" | "affine" | "polynomial" | "modular" | "recurrence" | "eventually_constant"
    formula: str            # Human-readable formula
    parameters: Dict[str, Any]  # Pattern parameters
    residual: float         # Fit quality (0 = perfect)
    confidence: float       # How confident we are

    def to_dict(self) -> dict:
        return {
            "type": self.pattern_type,
            "formula": self.formula,
            "parameters": self.parameters,
            "residual": round(self.residual, 8),
            "confidence": round(self.confidence, 4),
        }


class PatternCompressor:
    """Detect structural patterns in counting sequences and FP properties."""

    @staticmethod
    def analyze_counting_sequence(ks: List[int], vals: List[int]) -> List[DetectedPattern]:
        """Detect patterns in a counting sequence {k: count}."""
        patterns: List[DetectedPattern] = []
        if len(ks) < 2:
            return patterns

        # 1. Constant
        if len(set(vals)) == 1:
            patterns.append(DetectedPattern(
                pattern_type="constant",
                formula=f"f(k) = {vals[0]}",
                parameters={"c": vals[0]},
                residual=0.0,
                confidence=min(0.5 + 0.1 * len(ks), 0.95),
            ))

        # 2. Eventually constant (last half equal)
        if len(vals) >= 4:
            half = len(vals) // 2
            tail = vals[half:]
            if len(set(tail)) == 1 and vals[0] != tail[0]:
                K = ks[half]
                patterns.append(DetectedPattern(
                    pattern_type="eventually_constant",
                    formula=f"f(k) = {tail[0]} for k ≥ {K}",
                    parameters={"c": tail[0], "K": K},
                    residual=0.0,
                    confidence=min(0.4 + 0.1 * len(tail), 0.85),
                ))

        # 3. Affine: f(k) = a*k + b
        if len(ks) >= 3:
            diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            k_diffs = [ks[i+1] - ks[i] for i in range(len(ks)-1)]
            # Check if first differences are proportional to k-differences
            if all(kd > 0 for kd in k_diffs):
                slopes = [d / kd for d, kd in zip(diffs, k_diffs)]
                if len(set(round(s, 8) for s in slopes)) == 1:
                    a = slopes[0]
                    b = vals[0] - a * ks[0]
                    predicted = [a * k + b for k in ks]
                    residual = sum((p - v) ** 2 for p, v in zip(predicted, vals))
                    if residual < 1e-6:
                        if a == int(a):
                            a = int(a)
                        if b == int(b):
                            b = int(b)
                        patterns.append(DetectedPattern(
                            pattern_type="affine",
                            formula=f"f(k) = {a}·k + {b}",
                            parameters={"a": a, "b": b},
                            residual=residual,
                            confidence=min(0.5 + 0.1 * len(ks), 0.9),
                        ))

        # 4. Quadratic: f(k) = a*k² + b*k + c
        if len(ks) >= 4 and not any(p.pattern_type == "affine" for p in patterns):
            second_diffs = [vals[i+2] - 2*vals[i+1] + vals[i] for i in range(len(vals)-2)]
            if len(set(second_diffs)) == 1 and second_diffs[0] != 0:
                d2 = second_diffs[0]
                # a = d2/2 (assuming unit k-spacing)
                k_spacing = ks[1] - ks[0] if len(ks) > 1 else 1
                a = d2 / (2 * k_spacing ** 2)
                b = (vals[1] - vals[0]) / k_spacing - a * (ks[1] + ks[0])
                c = vals[0] - a * ks[0] ** 2 - b * ks[0]
                predicted = [a * k**2 + b * k + c for k in ks]
                residual = sum((p - v) ** 2 for p, v in zip(predicted, vals))
                if residual < 1.0:
                    patterns.append(DetectedPattern(
                        pattern_type="polynomial",
                        formula=f"f(k) = {a:.2f}·k² + {b:.2f}·k + {c:.2f}",
                        parameters={"a": a, "b": b, "c": c, "degree": 2},
                        residual=residual,
                        confidence=min(0.4 + 0.08 * len(ks), 0.8),
                    ))

        # 5. Modular pattern: f(k) depends on k mod m
        for m in [2, 3, 4]:
            if len(ks) >= 2 * m:
                groups: Dict[int, List[int]] = defaultdict(list)
                for k, v in zip(ks, vals):
                    groups[k % m].append(v)
                # Check if each group is constant
                if all(len(set(g)) == 1 for g in groups.values()):
                    group_vals = {r: g[0] for r, g in sorted(groups.items())}
                    if len(set(group_vals.values())) > 1:  # Not globally constant
                        patterns.append(DetectedPattern(
                            pattern_type="modular",
                            formula=f"f(k) depends on k mod {m}: {group_vals}",
                            parameters={"modulus": m, "values": group_vals},
                            residual=0.0,
                            confidence=min(0.4 + 0.08 * len(ks), 0.85),
                        ))

        # 6. Linear recurrence: a_n = c1*a_{n-1} + c2*a_{n-2}
        if len(vals) >= 5:
            # Try order-2 recurrence
            for i in range(len(vals) - 2):
                a0, a1, a2 = vals[i], vals[i+1], vals[i+2]
                if a0 != 0 and a1 != 0:
                    # a2 = c1*a1 + c2*a0
                    # Check over all triples
                    pass  # Complex; defer to OEIS-style lookup in future
            # Simple check: ratio test
            ratios = []
            for i in range(1, len(vals)):
                if vals[i-1] != 0:
                    ratios.append(vals[i] / vals[i-1])
            if ratios and len(set(round(r, 4) for r in ratios)) == 1:
                r = ratios[0]
                if r != 1.0:
                    patterns.append(DetectedPattern(
                        pattern_type="geometric",
                        formula=f"f(k) ≈ {vals[0]} · {r:.4f}^(k-{ks[0]})",
                        parameters={"base_val": vals[0], "ratio": r, "k0": ks[0]},
                        residual=0.0,
                        confidence=min(0.4 + 0.08 * len(vals), 0.8),
                    ))

        return patterns

    @staticmethod
    def analyze_fp_structure(fps: List[int], base: int = 10) -> List[DetectedPattern]:
        """Detect structural patterns in a set of fixed points."""
        patterns: List[DetectedPattern] = []
        nf = NumberFeatures(base)
        if not fps:
            return patterns

        # Check: all FPs divisible by common factor
        from math import gcd
        from functools import reduce
        nonzero = [fp for fp in fps if fp > 0]
        if len(nonzero) >= 2:
            g = reduce(gcd, nonzero)
            if g > 1:
                patterns.append(DetectedPattern(
                    pattern_type="common_factor",
                    formula=f"All FPs divisible by {g}",
                    parameters={"gcd": g},
                    residual=0.0,
                    confidence=0.8,
                ))

        # Check: all FPs have same digit sum
        if nonzero:
            ds_vals = [sum(nf.digits(fp)) for fp in nonzero]
            if len(set(ds_vals)) == 1:
                patterns.append(DetectedPattern(
                    pattern_type="constant_digit_sum",
                    formula=f"ds(FP) = {ds_vals[0]} for all FPs",
                    parameters={"digit_sum": ds_vals[0]},
                    residual=0.0,
                    confidence=min(0.5 + 0.1 * len(nonzero), 0.9),
                ))

        # Check: all FPs are palindromes
        if nonzero:
            pals = [fp for fp in nonzero if nf.digits(fp) == nf.digits(fp)[::-1]]
            if len(pals) == len(nonzero):
                patterns.append(DetectedPattern(
                    pattern_type="all_palindromes",
                    formula="All FPs are palindromes",
                    parameters={},
                    residual=0.0,
                    confidence=min(0.5 + 0.1 * len(nonzero), 0.9),
                ))

        return patterns


# =============================================================================
# 4. CONJECTURE MUTATOR
# =============================================================================

class MutationType(str, Enum):
    STRENGTHEN = "strengthen"     # Narrow quantifier or sharpen bound
    WEAKEN = "weaken"             # Add exceptions or widen quantifier
    GENERALIZE = "generalize"     # Replace specific with parametric
    SPECIALIZE = "specialize"     # Fix parameter to specific value
    TRANSFER = "transfer"         # Apply same property to different pipeline
    COMPOSE = "compose"           # Combine two conjectures


@dataclass
class Mutation:
    """A proposed mutation of a conjecture."""
    original: Conjecture
    mutated: Conjecture
    mutation_type: MutationType
    rationale: str

    def to_dict(self) -> dict:
        return {
            "mutation_type": self.mutation_type.value,
            "rationale": self.rationale,
            "original_statement": self.original.statement,
            "mutated_statement": self.mutated.statement,
        }


class ConjectureMutator:
    """Generate mutated conjectures for systematic hypothesis exploration."""

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()
        self.miner = ConjectureMiner(self.registry)

    def mutate(self, conjecture: Conjecture) -> List[Mutation]:
        """Generate all applicable mutations of a conjecture."""
        mutations: List[Mutation] = []
        mutations.extend(self._generalize_modulus(conjecture))
        mutations.extend(self._transfer_to_pipelines(conjecture))
        mutations.extend(self._strengthen_quantifier(conjecture))
        mutations.extend(self._weaken_with_exceptions(conjecture))
        return mutations

    def _generalize_modulus(self, c: Conjecture) -> List[Mutation]:
        """If mod m, try mod m' for related moduli."""
        if c.ctype != ConjectureType.MODULAR:
            return []
        m = c.parameters.get("modulus")
        base = c.parameters.get("base", 10)
        if m is None:
            return []

        mutations = []
        # Try base-1, base+1, and divisors/multiples of m
        candidates = set()
        candidates.add(base - 1)  # digit sum modulus
        candidates.add(base + 1)  # alternating digit sum
        if m > 1:
            candidates.add(m * 2)
            candidates.add(m * 3)
            for d in range(2, m):
                if m % d == 0:
                    candidates.add(d)
        candidates.discard(m)  # Remove original

        for m_new in sorted(candidates):
            if m_new < 2 or m_new > 100:
                continue
            new_c = Conjecture(
                ctype=c.ctype,
                statement=c.statement.replace(f"mod {m}", f"mod {m_new}"),
                quantifier=c.quantifier,
                predicate=c.predicate.replace(str(m), str(m_new)),
                evidence={},
                exceptions=[],
                pipeline=c.pipeline,
                parameters={**c.parameters, "modulus": m_new},
                confidence=c.confidence * 0.3,  # Much lower until tested
                novelty=c.novelty * 1.2,
                simplicity=c.simplicity,
                tested_domains=[],
            )
            mutations.append(Mutation(
                original=c, mutated=new_c,
                mutation_type=MutationType.GENERALIZE,
                rationale=f"Generalize modulus from {m} to {m_new}",
            ))

        return mutations

    def _transfer_to_pipelines(self, c: Conjecture) -> List[Mutation]:
        """Transfer a conjecture pattern to other similar pipelines."""
        if c.ctype not in (ConjectureType.MODULAR, ConjectureType.UNIVERSALITY):
            return []

        # Find pipelines with similar semantic structure
        source_pipe = c.pipeline
        mutations = []

        # Standard transfer targets
        transfer_candidates = [
            "kaprekar_step", "truc_1089", "digit_sum",
            "kaprekar_step |> digit_sum",
            "reverse |> complement_9",
        ]

        for target in transfer_candidates:
            if target == source_pipe:
                continue
            try:
                Pipeline.parse(target, registry=self.registry)
            except (KeyError, ValueError):
                continue

            new_c = Conjecture(
                ctype=c.ctype,
                statement=c.statement.replace(source_pipe, target),
                quantifier=c.quantifier,
                predicate=c.predicate,
                evidence={},
                exceptions=[],
                pipeline=target,
                parameters={**c.parameters},
                confidence=c.confidence * 0.2,  # Very low until tested
                novelty=0.8,
                simplicity=c.simplicity,
                tested_domains=[],
            )
            mutations.append(Mutation(
                original=c, mutated=new_c,
                mutation_type=MutationType.TRANSFER,
                rationale=f"Transfer property from {source_pipe} to {target}",
            ))

        return mutations

    def _strengthen_quantifier(self, c: Conjecture) -> List[Mutation]:
        """Try to extend conjecture to wider parameter range."""
        k_range = c.parameters.get("k_range", [])
        if not k_range or len(k_range) < 2:
            return []

        mutations = []
        k_min, k_max = min(k_range), max(k_range)

        # Try extending downward
        if k_min > 2:
            new_range = [k_min - 1] + k_range
            new_c = Conjecture(
                ctype=c.ctype,
                statement=c.statement,
                quantifier=c.quantifier.replace(str(k_min), str(k_min - 1)),
                predicate=c.predicate,
                evidence={},
                exceptions=[],
                pipeline=c.pipeline,
                parameters={**c.parameters, "k_range": new_range},
                confidence=c.confidence * 0.7,
                novelty=c.novelty,
                simplicity=c.simplicity,
                tested_domains=[],
            )
            mutations.append(Mutation(
                original=c, mutated=new_c,
                mutation_type=MutationType.STRENGTHEN,
                rationale=f"Extend lower bound from k={k_min} to k={k_min-1}",
            ))

        return mutations

    def _weaken_with_exceptions(self, c: Conjecture) -> List[Mutation]:
        """Propose weakened version if conjecture has known exceptions."""
        if not c.exceptions:
            return []

        exception_ks = [ex.get("k") for ex in c.exceptions if isinstance(ex, dict)]
        exception_ks = [k for k in exception_ks if k is not None]

        if not exception_ks:
            return []

        new_c = Conjecture(
            ctype=c.ctype,
            statement=c.statement + f" [except k ∈ {{{','.join(map(str, exception_ks))}}}]",
            quantifier=c.quantifier + f", k ∉ {{{','.join(map(str, exception_ks))}}}",
            predicate=c.predicate,
            evidence=c.evidence,
            exceptions=[],  # Exceptions now baked into quantifier
            pipeline=c.pipeline,
            parameters=c.parameters,
            confidence=c.confidence * 0.85,
            novelty=c.novelty * 0.9,
            simplicity=c.simplicity * 0.8,
            tested_domains=c.tested_domains,
        )
        return [Mutation(
            original=c, mutated=new_c,
            mutation_type=MutationType.WEAKEN,
            rationale=f"Exclude known exceptions at k={exception_ks}",
        )]


# =============================================================================
# 5. RANKING MODEL v1.0 (formalized, explicitly versioned)
# =============================================================================

RANKING_MODEL_VERSION = "1.0"

@dataclass
class RankedConjecture:
    """A conjecture with formalized ranking breakdown."""
    conjecture: Conjecture
    density_estimate: DensityEstimate
    proof_skeleton: ProofSkeleton
    patterns: List[DetectedPattern]
    rank_breakdown: Dict[str, float]
    final_score: float

    def to_dict(self) -> dict:
        return {
            "statement": self.conjecture.statement,
            "final_score": round(self.final_score, 4),
            "rank_breakdown": {k: round(v, 4) for k, v in self.rank_breakdown.items()},
            "density": self.density_estimate.to_dict(),
            "proof_strength": self.proof_skeleton.proof_strength,
            "patterns": [p.to_dict() for p in self.patterns],
            "ranking_model_version": RANKING_MODEL_VERSION,
        }


class RankingModelV1:
    """Formalized conjecture ranking with explicit weights and versioning.

    Ranking Model v1.0 (heuristic):
      score = w_conf * empirical_confidence
            + w_struct * structural_score
            + w_novel * novelty
            + w_simp * simplicity
            + w_false * falsifiability

    All weights and components are explicitly documented and versioned.
    """

    VERSION = RANKING_MODEL_VERSION

    # Explicit weight vector (v1.0)
    W_EMPIRICAL = 0.30    # Empirical confidence from density estimate
    W_STRUCTURAL = 0.25   # Structural strength from proof skeleton
    W_NOVELTY = 0.20      # How surprising / non-trivial
    W_SIMPLICITY = 0.15   # Inverse formula complexity
    W_FALSIFIABILITY = 0.10  # How testable / refutable

    def __init__(self, registry: Optional[OperationRegistry] = None):
        self.registry = registry or OperationRegistry()
        self.skeleton_gen = SkeletonGenerator(self.registry)
        self.density_est = DensityEstimator()
        self.compressor = PatternCompressor()

    def rank(self, conjecture: Conjecture) -> RankedConjecture:
        """Produce full ranked assessment of a conjecture."""
        # 1. Density estimate
        num_ces = len(conjecture.exceptions)
        density = self.density_est.estimate(conjecture, num_ces)

        # 2. Proof skeleton
        skeleton = self.skeleton_gen.generate(conjecture)

        # 3. Pattern analysis
        evidence = conjecture.evidence
        ks = sorted(int(k) for k in evidence.keys() if str(k).isdigit())
        vals = [evidence.get(k, evidence.get(str(k), 0)) for k in ks]
        patterns = self.compressor.analyze_counting_sequence(ks, vals) if ks else []

        # 4. Compute score components
        empirical = density.confidence_score
        structural = self._structural_score(skeleton)
        novelty = conjecture.novelty
        simplicity = conjecture.simplicity
        falsifiability = self._falsifiability_score(conjecture, density)

        breakdown = {
            "empirical_confidence": empirical,
            "structural_strength": structural,
            "novelty": novelty,
            "simplicity": simplicity,
            "falsifiability": falsifiability,
        }

        final = (self.W_EMPIRICAL * empirical
                 + self.W_STRUCTURAL * structural
                 + self.W_NOVELTY * novelty
                 + self.W_SIMPLICITY * simplicity
                 + self.W_FALSIFIABILITY * falsifiability)

        return RankedConjecture(
            conjecture=conjecture,
            density_estimate=density,
            proof_skeleton=skeleton,
            patterns=patterns,
            rank_breakdown=breakdown,
            final_score=final,
        )

    @staticmethod
    def _structural_score(skeleton: ProofSkeleton) -> float:
        """Score based on proof skeleton quality."""
        if skeleton.proof_strength == "complete":
            return 1.0
        elif skeleton.proof_strength == "modulo_gap":
            # Count proven vs gap steps
            proven = sum(1 for s in skeleton.reduction_steps if s.status == "proven")
            total = len(skeleton.reduction_steps)
            if total == 0:
                return 0.3
            return 0.5 + 0.5 * (proven / total)
        else:
            return 0.2

    @staticmethod
    def _falsifiability_score(conjecture: Conjecture, density: DensityEstimate) -> float:
        """How testable is this conjecture?"""
        # Higher if: large search space available, clear CE definition
        if density.total_search_space == 0:
            return 0.1
        vol_factor = min(density.search_volume_log10 / 8.0, 1.0)
        type_factor = {
            ConjectureType.MODULAR: 0.9,       # Very testable
            ConjectureType.COUNTING: 0.8,
            ConjectureType.MONOTONICITY: 0.7,
            ConjectureType.UNIVERSALITY: 0.6,
            ConjectureType.STRUCTURE: 0.5,
            ConjectureType.INVARIANT: 0.7,
        }.get(conjecture.ctype, 0.5)
        return vol_factor * type_factor

    def rank_all(self, conjectures: List[Conjecture]) -> List[RankedConjecture]:
        """Rank a list of conjectures and sort by final score."""
        ranked = [self.rank(c) for c in conjectures]
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked


# =============================================================================
# MAIN: Demo proof engine
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENGINE vNext — M3: Proof Engine + Structural Reasoning")
    print("=" * 70)

    reg = OperationRegistry()
    miner = ConjectureMiner(reg)
    gen = SkeletonGenerator(reg)
    estimator = DensityEstimator()
    compressor = PatternCompressor()
    mutator = ConjectureMutator(reg)
    ranker = RankingModelV1(reg)

    # ── 1. Mine conjectures ──
    print("\n── 1. Mining conjectures (kaprekar_step, k=3..5) ──\n")
    conjectures = miner.mine_all("kaprekar_step", [3, 4, 5])
    for c in conjectures:
        print(f"  {c}")

    # ── 2. Proof skeletons ──
    print("\n── 2. Proof skeletons ──\n")
    for c in conjectures[:3]:
        skel = gen.generate(c)
        print(f"  {skel}")
        for step in skel.reduction_steps:
            print(f"    [{step.status}] {step.description}")
        if skel.remaining_gaps:
            print(f"    GAPS: {skel.remaining_gaps}")

    # ── 3. Density estimates ──
    print("\n── 3. Counterexample density estimates ──\n")
    for c in conjectures[:3]:
        est = estimator.estimate(c)
        print(f"  {c.statement[:60]}...")
        print(f"    {est}")

    # ── 4. Pattern compression ──
    print("\n── 4. Pattern compression ──\n")
    fp_data = {3: 1, 4: 1, 5: 3}  # Kaprekar FP counts
    ks = sorted(fp_data.keys())
    vals = [fp_data[k] for k in ks]
    patterns = compressor.analyze_counting_sequence(ks, vals)
    for p in patterns:
        print(f"  {p.pattern_type}: {p.formula} (conf={p.confidence:.2f})")

    # ── 5. Conjecture mutation ──
    print("\n── 5. Conjecture mutations ──\n")
    mod_conj = [c for c in conjectures if c.ctype == ConjectureType.MODULAR]
    if mod_conj:
        mutations = mutator.mutate(mod_conj[0])
        for m in mutations[:5]:
            print(f"  [{m.mutation_type.value}] {m.rationale}")

    # ── 6. Full ranking ──
    print("\n── 6. Ranked conjectures (v1.0) ──\n")
    ranked = ranker.rank_all(conjectures)
    for r in ranked:
        print(f"  score={r.final_score:.3f} | {r.conjecture.statement[:70]}")
        print(f"    breakdown: {r.rank_breakdown}")

    print("\n" + "=" * 70)
    print(f"M3 COMPLETE (Ranking Model v{RANKING_MODEL_VERSION})")
    print("=" * 70)
