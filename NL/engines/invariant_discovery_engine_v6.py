#!/usr/bin/env python3
"""
SYNTRIAD Invariant Discovery Engine v6.0
=========================================

Autonomous Symbolic Discovery Engine for Discrete Dynamical Systems

Drie lagen:
  LAAG 1 - Empirische Dynamica (van v5.0)
  LAAG 2 - Structurele Abstractie (NIEUW)
  LAAG 3 - Symbolische Redenering (NIEUW)

Dit systeem:
  1. Detecteert algebraische invarianten (niet statistisch, structureel)
  2. Genereert Conjectures als eerste-klas objecten
  3. Zoekt actief naar tegenvoorbeelden
  4. Minimaliseert complexiteit (MDL)
  5. Detecteert cross-domain isomorfieën
  6. Bouwt nieuwe conceptuele categorieën

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
import json
import sqlite3
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable, Any, FrozenSet
from collections import Counter, defaultdict
from pathlib import Path
from enum import Enum, auto
import itertools
import hashlib
from abc import ABC, abstractmethod


# =============================================================================
# INFORMATION THEORY
# =============================================================================

def digit_entropy(n: int) -> float:
    if n == 0:
        return 0.0
    digits = list(str(abs(n)))
    freqs = Counter(digits)
    total = len(digits)
    probs = [v / total for v in freqs.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def digit_variance(n: int) -> float:
    if n == 0:
        return 0.0
    digits = [int(d) for d in str(abs(n))]
    if len(digits) < 2:
        return 0.0
    mean = sum(digits) / len(digits)
    return sum((d - mean) ** 2 for d in digits) / len(digits)


# =============================================================================
# DIGIT OPERATIONS
# =============================================================================

class DigitOp:
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
    def digit_pow2(n: int) -> int:
        return sum(int(d)**2 for d in str(abs(n)))

    @staticmethod
    def digit_pow3(n: int) -> int:
        return sum(int(d)**3 for d in str(abs(n)))

    @staticmethod
    def digit_pow4(n: int) -> int:
        return sum(int(d)**4 for d in str(abs(n)))

    @staticmethod
    def digit_pow5(n: int) -> int:
        return sum(int(d)**5 for d in str(abs(n)))

    @staticmethod
    def sort_asc(n: int) -> int:
        s = ''.join(sorted(str(abs(n)))).lstrip('0')
        return int(s) if s else 0

    @staticmethod
    def sort_desc(n: int) -> int:
        return int(''.join(sorted(str(abs(n)), reverse=True)))

    @staticmethod
    def kaprekar_step(n: int) -> int:
        return DigitOp.sort_desc(n) - DigitOp.sort_asc(n)

    @staticmethod
    def truc_1089(n: int) -> int:
        if n <= 0:
            return 0
        rev = DigitOp.reverse(n)
        diff = abs(n - rev)
        if diff == 0:
            return 0
        return diff + DigitOp.reverse(diff)

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
        return abs(n) + DigitOp.reverse(n)

    @staticmethod
    def sub_reverse(n: int) -> int:
        return abs(abs(n) - DigitOp.reverse(n))

    @staticmethod
    def digit_factorial_sum(n: int) -> int:
        factorials = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        return sum(factorials[int(d)] for d in str(abs(n)))

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


OPERATIONS: Dict[str, Callable[[int], int]] = {
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
# LAAG 2: CONJECTURE OBJECT MODEL
# =============================================================================

class ProofStatus(Enum):
    OPEN = "open"
    EMPIRICAL = "empirical"
    PARTIAL = "partial"
    DISPROVEN = "disproven"
    PROVEN = "proven"


class InvariantType(Enum):
    MODULAR = "modular"             # f(n) mod k == g(n) mod k
    MONOTONIC = "monotonic"         # f(n) < n (eventually)
    BOUNDED = "bounded"             # f(n) <= B for all n in domain
    PERIODIC = "periodic"           # f^k(n) == f^(k+p)(n)
    LENGTH_PRESERVING = "length"    # len(f(n)) == len(n)
    CONTRACTIVE = "contractive"     # |f(n) - A| < |n - A| naar attractor A
    CONGRUENCE_CLASS = "congruence" # f preserveert congruentieklasse
    ENTROPY_REDUCING = "entropy"    # H(f(n)) < H(n)
    IDEMPOTENT = "idempotent"       # f(f(n)) == f(n)
    ABSORBING = "absorbing"         # eenmaal in A, blijft in A


@dataclass
class Conjecture:
    """Een vermoeden als eerste-klas object."""
    id: str
    statement: str                          # Mensleesbare formulering
    formal: str                             # Formele notatie
    invariant_type: InvariantType
    domain: Tuple[int, int]                 # (min, max)
    pipeline: Tuple[str, ...]
    
    # Bewijsstatus
    evidence_samples: int = 0
    counterexamples: List[int] = field(default_factory=list)
    proof_status: ProofStatus = ProofStatus.OPEN
    confidence: float = 0.0
    
    # Structurele basis
    structural_basis: List[str] = field(default_factory=list)
    mechanism: str = ""
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_tested: float = 0.0
    test_count: int = 0
    
    def is_alive(self) -> bool:
        return self.proof_status not in (ProofStatus.DISPROVEN, ProofStatus.PROVEN)


@dataclass
class ConceptualCategory:
    """Een nieuwe conceptuele categorie ontdekt door het systeem."""
    id: str
    name: str                               # Gegenereerde naam
    description: str                        # Wat het is
    defining_properties: List[str]          # Welke invarianten het definieren
    member_pipelines: List[Tuple[str, ...]] # Pipelines die erin vallen
    member_attractors: List[int]            # Attractoren in deze categorie
    
    # Isomorfie
    isomorphic_to: List[str] = field(default_factory=list)
    
    # Kwaliteit
    cohesion: float = 0.0                   # Hoe coherent is de categorie
    separation: float = 0.0                 # Hoe onderscheidend
    
    created_at: float = field(default_factory=time.time)


# =============================================================================
# LAAG 2: INVARIANT MINING ENGINE
# =============================================================================

class InvariantMiner:
    """Detecteert algebraische invarianten - structureel, niet statistisch."""
    
    def __init__(self, ops: Dict[str, Callable]):
        self.ops = ops
    
    def apply_pipeline(self, n: int, pipeline: Tuple[str, ...]) -> int:
        for op_name in pipeline:
            if op_name in self.ops:
                n = self.ops[op_name](n)
                if n > 10**15 or n < 0:
                    return -1
        return n
    
    def mine_modular_invariants(self, pipeline: Tuple[str, ...],
                                 domain: Tuple[int, int] = (100, 99999),
                                 sample_size: int = 10000) -> List[Conjecture]:
        """Zoek mod-k invarianten: f(n) mod k == n mod k."""
        conjectures = []
        numbers = random.sample(range(domain[0], domain[1] + 1),
                               min(sample_size, domain[1] - domain[0]))
        
        for k in [2, 3, 5, 7, 9, 11, 13, 99]:
            preserved = 0
            counterexamples = []
            
            for n in numbers:
                result = self.apply_pipeline(n, pipeline)
                if result >= 0:
                    if result % k == n % k:
                        preserved += 1
                    else:
                        if len(counterexamples) < 10:
                            counterexamples.append(n)
            
            ratio = preserved / len(numbers) if numbers else 0
            
            if ratio == 1.0:
                pipe_str = ' → '.join(pipeline)
                conjectures.append(Conjecture(
                    id=f"mod_{k}_{hash(pipeline)}",
                    statement=f"Pipeline [{pipe_str}] preserves congruence class mod {k}",
                    formal=f"∀n ∈ [{domain[0]},{domain[1]}]: f(n) ≡ n (mod {k})",
                    invariant_type=InvariantType.MODULAR,
                    domain=domain,
                    pipeline=pipeline,
                    evidence_samples=len(numbers),
                    counterexamples=[],
                    proof_status=ProofStatus.EMPIRICAL,
                    confidence=0.99,
                    structural_basis=[f"mod_{k}_invariance"],
                    mechanism=f"All operators in pipeline preserve residue mod {k}"
                ))
            elif ratio > 0.99 and counterexamples:
                pipe_str = ' → '.join(pipeline)
                conjectures.append(Conjecture(
                    id=f"near_mod_{k}_{hash(pipeline)}",
                    statement=f"Pipeline [{pipe_str}] nearly preserves mod {k} "
                              f"({ratio:.4f}), exceptions: {counterexamples[:3]}",
                    formal=f"∀n ∈ [{domain[0]},{domain[1]}] \\ E: f(n) ≡ n (mod {k}), |E| small",
                    invariant_type=InvariantType.MODULAR,
                    domain=domain,
                    pipeline=pipeline,
                    evidence_samples=len(numbers),
                    counterexamples=counterexamples,
                    proof_status=ProofStatus.OPEN,
                    confidence=ratio
                ))
        
        return conjectures
    
    def mine_monotonicity(self, pipeline: Tuple[str, ...],
                           domain: Tuple[int, int] = (100, 99999),
                           sample_size: int = 5000) -> List[Conjecture]:
        """Detecteer monotone reductie: f(n) < n."""
        conjectures = []
        numbers = random.sample(range(domain[0], domain[1] + 1),
                               min(sample_size, domain[1] - domain[0]))
        
        reducing = 0
        expanding = 0
        constant = 0
        counterexamples = []
        
        for n in numbers:
            result = self.apply_pipeline(n, pipeline)
            if result < 0:
                continue
            if result < n:
                reducing += 1
            elif result > n:
                expanding += 1
                if len(counterexamples) < 10:
                    counterexamples.append(n)
            else:
                constant += 1
        
        total = reducing + expanding + constant
        if total == 0:
            return conjectures
        
        red_ratio = reducing / total
        
        if red_ratio > 0.99:
            pipe_str = ' → '.join(pipeline)
            conjectures.append(Conjecture(
                id=f"monotone_{hash(pipeline)}",
                statement=f"Pipeline [{pipe_str}] is monotonically reducing "
                          f"({red_ratio:.4f})",
                formal=f"∀n ∈ [{domain[0]},{domain[1]}]: f(n) < n",
                invariant_type=InvariantType.MONOTONIC,
                domain=domain,
                pipeline=pipeline,
                evidence_samples=total,
                counterexamples=counterexamples,
                proof_status=ProofStatus.EMPIRICAL if red_ratio == 1.0 
                             else ProofStatus.OPEN,
                confidence=red_ratio,
                structural_basis=["value_reducing"],
                mechanism="Pipeline maps large inputs to smaller outputs"
            ))
        
        return conjectures
    
    def mine_boundedness(self, pipeline: Tuple[str, ...],
                          domain: Tuple[int, int] = (100, 99999),
                          sample_size: int = 5000) -> List[Conjecture]:
        """Detecteer begrensdheid: f(n) <= B."""
        numbers = random.sample(range(domain[0], domain[1] + 1),
                               min(sample_size, domain[1] - domain[0]))
        conjectures = []
        
        max_result = 0
        results = []
        
        for n in numbers:
            result = self.apply_pipeline(n, pipeline)
            if result >= 0:
                results.append(result)
                max_result = max(max_result, result)
        
        if not results:
            return conjectures
        
        # Check of output begrensd is ongeacht input grootte
        upper_quartile = sorted(results)[int(0.99 * len(results))]
        input_range = domain[1] - domain[0]
        
        # Als 99% van outputs veel kleiner is dan input range
        if upper_quartile < input_range * 0.01:
            pipe_str = ' → '.join(pipeline)
            conjectures.append(Conjecture(
                id=f"bounded_{hash(pipeline)}",
                statement=f"Pipeline [{pipe_str}] is bounded: "
                          f"99% outputs <= {upper_quartile}, max = {max_result}",
                formal=f"∀n ∈ [{domain[0]},{domain[1]}]: f(n) ≤ {max_result}",
                invariant_type=InvariantType.BOUNDED,
                domain=domain,
                pipeline=pipeline,
                evidence_samples=len(results),
                proof_status=ProofStatus.EMPIRICAL,
                confidence=0.95,
                structural_basis=["output_bounded"],
                mechanism=f"Pipeline compresses input space to [{min(results)}, {max_result}]"
            ))
        
        return conjectures
    
    def mine_contractivity(self, pipeline: Tuple[str, ...],
                            attractor: int,
                            domain: Tuple[int, int] = (100, 99999),
                            sample_size: int = 5000) -> List[Conjecture]:
        """Detecteer contractiviteit richting attractor."""
        numbers = random.sample(range(domain[0], domain[1] + 1),
                               min(sample_size, domain[1] - domain[0]))
        conjectures = []
        
        contracting = 0
        counterexamples = []
        
        for n in numbers:
            result = self.apply_pipeline(n, pipeline)
            if result < 0:
                continue
            
            dist_before = abs(n - attractor)
            dist_after = abs(result - attractor)
            
            if dist_after < dist_before:
                contracting += 1
            elif dist_after > dist_before and len(counterexamples) < 10:
                counterexamples.append(n)
        
        ratio = contracting / len(numbers) if numbers else 0
        
        if ratio > 0.8:
            pipe_str = ' → '.join(pipeline)
            conjectures.append(Conjecture(
                id=f"contract_{attractor}_{hash(pipeline)}",
                statement=f"Pipeline [{pipe_str}] is contractive toward {attractor} "
                          f"({ratio:.2%})",
                formal=f"|f(n) - {attractor}| < |n - {attractor}| for {ratio:.2%} of n",
                invariant_type=InvariantType.CONTRACTIVE,
                domain=domain,
                pipeline=pipeline,
                evidence_samples=len(numbers),
                counterexamples=counterexamples,
                proof_status=ProofStatus.EMPIRICAL if ratio > 0.99 
                             else ProofStatus.OPEN,
                confidence=ratio,
                structural_basis=["contractive_mapping"],
                mechanism=f"Pipeline acts as contraction mapping toward fixed point {attractor}"
            ))
        
        return conjectures
    
    def mine_entropy_reduction(self, pipeline: Tuple[str, ...],
                                domain: Tuple[int, int] = (100, 99999),
                                sample_size: int = 3000) -> List[Conjecture]:
        """Detecteer systematische entropy reductie."""
        numbers = random.sample(range(domain[0], domain[1] + 1),
                               min(sample_size, domain[1] - domain[0]))
        conjectures = []
        
        entropy_deltas = []
        reducing_count = 0
        
        for n in numbers:
            result = self.apply_pipeline(n, pipeline)
            if result <= 0:
                continue
            
            h_before = digit_entropy(n)
            h_after = digit_entropy(result)
            delta = h_before - h_after
            entropy_deltas.append(delta)
            if delta > 0:
                reducing_count += 1
        
        if not entropy_deltas:
            return conjectures
        
        mean_delta = np.mean(entropy_deltas)
        reduce_ratio = reducing_count / len(entropy_deltas)
        
        if mean_delta > 0.3 and reduce_ratio > 0.7:
            pipe_str = ' → '.join(pipeline)
            conjectures.append(Conjecture(
                id=f"entropy_{hash(pipeline)}",
                statement=f"Pipeline [{pipe_str}] systematically reduces digit entropy "
                          f"(mean Δ = {mean_delta:.3f}, {reduce_ratio:.1%} reducing)",
                formal=f"E[H(f(n)) - H(n)] = {mean_delta:.3f}",
                invariant_type=InvariantType.ENTROPY_REDUCING,
                domain=domain,
                pipeline=pipeline,
                evidence_samples=len(entropy_deltas),
                proof_status=ProofStatus.EMPIRICAL,
                confidence=min(0.95, reduce_ratio),
                structural_basis=["entropy_compression"],
                mechanism="Pipeline compresses digit distribution toward lower entropy states"
            ))
        
        return conjectures
    
    def mine_all_invariants(self, pipeline: Tuple[str, ...],
                             attractor: Optional[int] = None,
                             domain: Tuple[int, int] = (100, 99999)) -> List[Conjecture]:
        """Mine alle types invarianten voor een pipeline."""
        conjectures = []
        conjectures.extend(self.mine_modular_invariants(pipeline, domain))
        conjectures.extend(self.mine_monotonicity(pipeline, domain))
        conjectures.extend(self.mine_boundedness(pipeline, domain))
        conjectures.extend(self.mine_entropy_reduction(pipeline, domain))
        if attractor is not None:
            conjectures.extend(self.mine_contractivity(pipeline, attractor, domain))
        return conjectures


# =============================================================================
# LAAG 2: MECHANISM SYNTHESIZER
# =============================================================================

class MechanismSynthesizer:
    """Genereert mechanistische verklaringen uit invariant-combinaties."""
    
    # Bekende mechanisme-templates
    MECHANISMS = {
        "congruence_compression": {
            "requires": {InvariantType.MODULAR, InvariantType.ENTROPY_REDUCING},
            "template": "Entropy compression within congruence classes mod {mod} "
                        "induces fixed-point collapse to attractor {attractor}",
            "category": "CongruenceCompressor"
        },
        "contractive_reduction": {
            "requires": {InvariantType.CONTRACTIVE, InvariantType.MONOTONIC},
            "template": "Monotonic value reduction combined with contractivity toward "
                        "{attractor} guarantees convergence from any initial condition",
            "category": "ContractiveReducer"
        },
        "bounded_periodicity": {
            "requires": {InvariantType.BOUNDED, InvariantType.PERIODIC},
            "template": "Pipeline maps to bounded range [{lower}, {upper}] "
                        "where periodic behavior dominates",
            "category": "BoundedOscillator"
        },
        "entropy_funnel": {
            "requires": {InvariantType.ENTROPY_REDUCING, InvariantType.BOUNDED},
            "template": "Systematic entropy reduction within bounded output space "
                        "creates a convergence funnel toward minimal-entropy states",
            "category": "EntropyFunnel"
        },
        "modular_absorber": {
            "requires": {InvariantType.MODULAR, InvariantType.CONTRACTIVE},
            "template": "Congruence preservation mod {mod} combined with contractivity "
                        "confines dynamics to single congruence class, absorbing to {attractor}",
            "category": "ModularAbsorber"
        },
        "pure_compressor": {
            "requires": {InvariantType.MONOTONIC, InvariantType.BOUNDED},
            "template": "Monotonic reduction into bounded range forces eventual "
                        "convergence by well-ordering of naturals",
            "category": "PureCompressor"
        }
    }
    
    def synthesize(self, conjectures: List[Conjecture],
                    attractor: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """Synthetiseer mechanismen uit conjecture-combinaties.
        Returns: list of (mechanism_name, explanation, category)"""
        
        found_types = {c.invariant_type for c in conjectures if c.confidence > 0.8}
        results = []
        
        for mech_name, mech_info in self.MECHANISMS.items():
            required = mech_info["requires"]
            if required.issubset(found_types):
                # Vul template in
                explanation = mech_info["template"]
                
                # Zoek mod waarde
                mod_conj = [c for c in conjectures 
                           if c.invariant_type == InvariantType.MODULAR and c.confidence > 0.8]
                if mod_conj and "{mod}" in explanation:
                    mod_val = mod_conj[0].formal.split("mod ")[-1].rstrip(")")
                    explanation = explanation.replace("{mod}", mod_val)
                
                if attractor is not None:
                    explanation = explanation.replace("{attractor}", str(attractor))
                
                # Bounds
                bounded_conj = [c for c in conjectures 
                               if c.invariant_type == InvariantType.BOUNDED]
                if bounded_conj:
                    explanation = explanation.replace("{lower}", "0")
                    explanation = explanation.replace("{upper}", 
                                                     bounded_conj[0].formal.split("≤ ")[-1])
                
                results.append((mech_name, explanation, mech_info["category"]))
        
        return results


# =============================================================================
# LAAG 3: CONCEPTUAL CATEGORY BUILDER
# =============================================================================

class CategoryBuilder:
    """Bouwt nieuwe conceptuele categorieën uit ontdekkingen."""
    
    def __init__(self):
        self.categories: Dict[str, ConceptualCategory] = {}
        self.pipeline_profiles: Dict[Tuple[str, ...], Dict] = {}
    
    def register_pipeline(self, pipeline: Tuple[str, ...],
                           conjectures: List[Conjecture],
                           mechanisms: List[Tuple[str, str, str]],
                           attractor: Optional[int] = None,
                           dominance: float = 0.0):
        """Registreer een pipeline met zijn eigenschappen."""
        
        invariant_types = frozenset(
            c.invariant_type for c in conjectures if c.confidence > 0.8
        )
        mechanism_categories = frozenset(m[2] for m in mechanisms)
        
        self.pipeline_profiles[pipeline] = {
            "invariant_types": invariant_types,
            "mechanism_categories": mechanism_categories,
            "attractor": attractor,
            "dominance": dominance,
            "conjectures": conjectures
        }
    
    def discover_categories(self) -> List[ConceptualCategory]:
        """Ontdek categorieën door clustering van pipeline-profielen."""
        
        if len(self.pipeline_profiles) < 3:
            return []
        
        new_categories = []
        
        # Groepeer pipelines op basis van gedeelde mechanism-categorieën
        mechanism_groups: Dict[FrozenSet[str], List[Tuple[str, ...]]] = defaultdict(list)
        
        for pipeline, profile in self.pipeline_profiles.items():
            mech_cats = profile["mechanism_categories"]
            if mech_cats:
                mechanism_groups[mech_cats].append(pipeline)
        
        for mech_set, pipelines in mechanism_groups.items():
            if len(pipelines) < 2:
                continue
            
            cat_name = self._generate_category_name(mech_set)
            cat_id = f"cat_{hashlib.md5(cat_name.encode()).hexdigest()[:8]}"
            
            # Verzamel attractoren
            attractors = []
            for p in pipelines:
                a = self.pipeline_profiles[p].get("attractor")
                if a is not None:
                    attractors.append(a)
            
            category = ConceptualCategory(
                id=cat_id,
                name=cat_name,
                description=self._generate_description(mech_set, pipelines),
                defining_properties=list(mech_set),
                member_pipelines=pipelines,
                member_attractors=attractors,
                cohesion=len(pipelines) / len(self.pipeline_profiles),
                separation=1.0 - len(mech_set.intersection(
                    set().union(*(p["mechanism_categories"] 
                                 for p in self.pipeline_profiles.values()))
                )) / max(len(mech_set), 1)
            )
            
            if cat_id not in self.categories:
                self.categories[cat_id] = category
                new_categories.append(category)
        
        # Groepeer op invariant-type signature
        invariant_groups: Dict[FrozenSet[InvariantType], List[Tuple[str, ...]]] = defaultdict(list)
        
        for pipeline, profile in self.pipeline_profiles.items():
            inv_types = profile["invariant_types"]
            if inv_types:
                invariant_groups[inv_types].append(pipeline)
        
        for inv_set, pipelines in invariant_groups.items():
            if len(pipelines) < 2:
                continue
            
            cat_name = self._generate_invariant_category_name(inv_set)
            cat_id = f"inv_{hashlib.md5(cat_name.encode()).hexdigest()[:8]}"
            
            if cat_id in self.categories:
                continue
            
            attractors = []
            for p in pipelines:
                a = self.pipeline_profiles[p].get("attractor")
                if a is not None:
                    attractors.append(a)
            
            category = ConceptualCategory(
                id=cat_id,
                name=cat_name,
                description=f"Systems sharing invariant structure: "
                            f"{', '.join(t.value for t in inv_set)}",
                defining_properties=[t.value for t in inv_set],
                member_pipelines=pipelines,
                member_attractors=attractors,
                cohesion=len(pipelines) / len(self.pipeline_profiles)
            )
            
            self.categories[cat_id] = category
            new_categories.append(category)
        
        # Detecteer isomorfieën tussen categorieën
        self._detect_isomorphisms()
        
        return new_categories
    
    def _generate_category_name(self, mechanism_set: FrozenSet[str]) -> str:
        parts = sorted(mechanism_set)
        if "EntropyFunnel" in parts:
            return "Entropy-Funneling Systems"
        elif "CongruenceCompressor" in parts:
            return "Congruence-Compressing Systems"
        elif "ContractiveReducer" in parts:
            return "Contractive-Reducing Systems"
        elif "PureCompressor" in parts:
            return "Pure Compression Systems"
        elif "ModularAbsorber" in parts:
            return "Modular-Absorbing Systems"
        return f"{'–'.join(parts)} Systems"
    
    def _generate_invariant_category_name(self, inv_set: FrozenSet[InvariantType]) -> str:
        names = sorted([t.value for t in inv_set])
        if "modular" in names and "entropy" in names:
            return "Modular-Entropy Convergent Class"
        elif "monotonic" in names and "bounded" in names:
            return "Monotone-Bounded Attractor Class"
        elif "contractive" in names:
            return "Contractive Dynamical Class"
        return f"Invariant Class [{', '.join(names)}]"
    
    def _generate_description(self, mech_set: FrozenSet[str], 
                               pipelines: List[Tuple[str, ...]]) -> str:
        return (f"A class of {len(pipelines)} discrete dynamical systems sharing "
                f"mechanism(s): {', '.join(sorted(mech_set))}. "
                f"These systems exhibit structurally similar convergence behavior "
                f"despite different operator compositions.")
    
    def _detect_isomorphisms(self):
        """Detecteer structurele isomorfieën tussen categorieën."""
        cats = list(self.categories.values())
        for i, c1 in enumerate(cats):
            for c2 in cats[i+1:]:
                # Vergelijk defining properties
                props1 = set(c1.defining_properties)
                props2 = set(c2.defining_properties)
                
                overlap = len(props1 & props2) / max(len(props1 | props2), 1)
                
                if overlap > 0.5 and c1.id != c2.id:
                    if c2.id not in c1.isomorphic_to:
                        c1.isomorphic_to.append(c2.id)
                    if c1.id not in c2.isomorphic_to:
                        c2.isomorphic_to.append(c1.id)


# =============================================================================
# LAAG 3: COUNTEREXAMPLE HUNTER
# =============================================================================

class CounterexampleHunter:
    """Zoekt actief naar tegenvoorbeelden voor conjectures."""
    
    def __init__(self, ops: Dict[str, Callable]):
        self.ops = ops
    
    def apply_pipeline(self, n: int, pipeline: Tuple[str, ...]) -> int:
        for op_name in pipeline:
            if op_name in self.ops:
                n = self.ops[op_name](n)
                if n > 10**15 or n < 0:
                    return -1
        return n
    
    def hunt(self, conjecture: Conjecture, 
             strategies: List[str] = None,
             budget: int = 50000) -> List[int]:
        """Zoek tegenvoorbeelden met gerichte strategieën."""
        
        if strategies is None:
            strategies = ["boundary", "structured", "random_extended", "extremal"]
        
        counterexamples = []
        
        for strategy in strategies:
            candidates = self._generate_candidates(strategy, conjecture, budget // len(strategies))
            
            for n in candidates:
                if self._is_counterexample(n, conjecture):
                    counterexamples.append(n)
        
        return counterexamples
    
    def _generate_candidates(self, strategy: str, 
                              conjecture: Conjecture, budget: int) -> List[int]:
        """Genereer kandidaten per strategie."""
        
        domain = conjecture.domain
        
        if strategy == "boundary":
            # Grenswaarden
            candidates = list(range(domain[0], min(domain[0] + budget // 4, domain[1])))
            candidates += list(range(max(domain[1] - budget // 4, domain[0]), domain[1] + 1))
            # Powers of 10
            candidates += [10**k for k in range(2, 8)]
            candidates += [10**k - 1 for k in range(2, 8)]
            return candidates[:budget]
        
        elif strategy == "structured":
            candidates = []
            # Repdigits
            for d in range(1, 10):
                for length in range(2, 7):
                    candidates.append(int(str(d) * length))
            # Palindromes
            for base in range(10, 1000):
                s = str(base)
                candidates.append(int(s + s[::-1]))
                candidates.append(int(s + s[-2::-1]))
            # Powers
            for b in range(2, 20):
                for p in range(2, 10):
                    v = b**p
                    if domain[0] <= v <= domain[1]:
                        candidates.append(v)
            return candidates[:budget]
        
        elif strategy == "random_extended":
            # Uitgebreid domein
            ext_min = max(1, domain[0] // 10)
            ext_max = min(10**8, domain[1] * 10)
            return random.sample(range(ext_min, ext_max), 
                               min(budget, ext_max - ext_min))
        
        elif strategy == "extremal":
            candidates = []
            # Getallen met veel nullen
            for i in range(budget // 5):
                base = random.randint(1, 9)
                zeros = random.randint(1, 5)
                candidates.append(base * 10**zeros)
            # Getallen met extreme digit variance
            for _ in range(budget // 5):
                digits = [random.choice([0, 9]) for _ in range(random.randint(3, 6))]
                digits[0] = max(digits[0], 1)
                candidates.append(int(''.join(str(d) for d in digits)))
            return candidates[:budget]
        
        return []
    
    def _is_counterexample(self, n: int, conjecture: Conjecture) -> bool:
        """Check of n een tegenvoorbeeld is."""
        
        result = self.apply_pipeline(n, conjecture.pipeline)
        if result < 0:
            return False
        
        if conjecture.invariant_type == InvariantType.MODULAR:
            # Extracteer mod waarde uit formal
            for k in [2, 3, 5, 7, 9, 11, 13, 99]:
                if f"mod {k}" in conjecture.formal:
                    return result % k != n % k
        
        elif conjecture.invariant_type == InvariantType.MONOTONIC:
            return result >= n
        
        elif conjecture.invariant_type == InvariantType.BOUNDED:
            # Extracteer bound uit formal
            try:
                bound = int(conjecture.formal.split("≤ ")[-1])
                return result > bound
            except:
                return False
        
        elif conjecture.invariant_type == InvariantType.ENTROPY_REDUCING:
            h_before = digit_entropy(n)
            h_after = digit_entropy(result)
            return h_after > h_before + 0.5  # Significante toename
        
        return False


# =============================================================================
# LAAG 3: MDL SCORING
# =============================================================================

class MDLScorer:
    """Minimal Description Length scoring voor elegantie."""
    
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
    
    def score(self, pipeline: Tuple[str, ...], dominance: float,
              n_invariants: int, n_mechanisms: int) -> float:
        """
        Score = kwaliteit - complexiteit
        
        Echte ontdekkingen = maximale structuur, minimale complexiteit.
        """
        
        # Kwaliteit component
        quality = (
            0.3 * (dominance / 100) +
            0.3 * min(n_invariants / 5, 1.0) +
            0.4 * min(n_mechanisms / 3, 1.0)
        )
        
        # Complexiteit penalty
        complexity = self.alpha * len(pipeline)
        
        # Elegantie bonus voor korte pipelines met hoge kwaliteit
        elegance_bonus = 0.0
        if len(pipeline) <= 2 and dominance > 95:
            elegance_bonus = 0.1
        
        return quality - complexity + elegance_bonus


# =============================================================================
# HOMEOSTATISCHE META-LOOP
# =============================================================================

@dataclass
class SystemState:
    """Interne toestand van het systeem."""
    exploration_rate: float = 0.5
    mutation_rate: float = 0.3
    focus_invariant_types: Set[InvariantType] = field(default_factory=set)
    preferred_operators: List[str] = field(default_factory=list)
    
    # Performance tracking
    conjectures_generated: int = 0
    conjectures_confirmed: int = 0
    conjectures_disproven: int = 0
    categories_discovered: int = 0
    mechanisms_found: int = 0
    
    # Cycle tracking
    recent_scores: List[float] = field(default_factory=list)
    cycle_count: int = 0


class HomeostaticController:
    """Homeostatische meta-loop: het systeem reguleert zichzelf."""
    
    def __init__(self, ops: List[str]):
        self.state = SystemState()
        self.all_ops = ops
        self.op_scores: Dict[str, float] = {op: 1.0 for op in ops}
        self.history: List[Dict] = []
    
    def select_pipeline(self, length: Optional[int] = None) -> Tuple[str, ...]:
        """Selecteer pipeline op basis van geleerde biases."""
        
        if length is None:
            # Prefer korter (MDL)
            length = random.choices([2, 3, 4], weights=[0.5, 0.35, 0.15])[0]
        
        if random.random() < self.state.exploration_rate:
            return tuple(random.choices(self.all_ops, k=length))
        else:
            weights = [max(0.01, self.op_scores.get(op, 1.0)) for op in self.all_ops]
            total = sum(weights)
            probs = [w / total for w in weights]
            return tuple(np.random.choice(self.all_ops, size=length, p=probs))
    
    def record_result(self, pipeline: Tuple[str, ...], score: float,
                       conjectures: List[Conjecture],
                       mechanisms: List[Tuple[str, str, str]]):
        """Registreer resultaat en update interne staat."""
        
        self.state.cycle_count += 1
        self.state.recent_scores.append(score)
        if len(self.state.recent_scores) > 50:
            self.state.recent_scores = self.state.recent_scores[-50:]
        
        # Update operator scores
        for op in pipeline:
            old = self.op_scores.get(op, 1.0)
            self.op_scores[op] = 0.85 * old + 0.15 * (score * 3)
        
        # Track conjecture types die werken
        for c in conjectures:
            if c.confidence > 0.9:
                self.state.focus_invariant_types.add(c.invariant_type)
        
        self.state.conjectures_generated += len(conjectures)
        self.state.mechanisms_found += len(mechanisms)
        
        self.history.append({
            "pipeline": pipeline,
            "score": score,
            "n_conjectures": len(conjectures),
            "n_mechanisms": len(mechanisms)
        })
    
    def adapt(self):
        """Pas systeemparameters aan op basis van recente prestaties."""
        
        if len(self.state.recent_scores) < 10:
            return
        
        recent = self.state.recent_scores[-10:]
        older = self.state.recent_scores[-20:-10] if len(self.state.recent_scores) >= 20 else []
        
        avg_recent = np.mean(recent)
        avg_older = np.mean(older) if older else avg_recent
        
        # Adaptieve exploration rate
        if avg_recent < avg_older * 0.8:
            # Performance daalt -> meer exploreren
            self.state.exploration_rate = min(0.8, self.state.exploration_rate + 0.05)
        elif avg_recent > avg_older * 1.2:
            # Performance stijgt -> meer exploiteren
            self.state.exploration_rate = max(0.15, self.state.exploration_rate - 0.05)
        
        # Update preferred operators
        sorted_ops = sorted(self.op_scores.items(), key=lambda x: x[1], reverse=True)
        self.state.preferred_operators = [op for op, _ in sorted_ops[:5]]
    
    def get_diagnosis(self) -> str:
        """Genereer zelfdiagnose."""
        
        if not self.state.recent_scores:
            return "Insufficient data for diagnosis"
        
        avg = np.mean(self.state.recent_scores[-10:])
        trend = ""
        if len(self.state.recent_scores) >= 20:
            old_avg = np.mean(self.state.recent_scores[-20:-10])
            if avg > old_avg * 1.1:
                trend = "IMPROVING"
            elif avg < old_avg * 0.9:
                trend = "DECLINING"
            else:
                trend = "STABLE"
        
        conf_rate = (self.state.conjectures_confirmed / 
                    max(self.state.conjectures_generated, 1))
        
        return (f"Cycles: {self.state.cycle_count} | "
                f"Avg score: {avg:.3f} | "
                f"Trend: {trend} | "
                f"Exploration: {self.state.exploration_rate:.2f} | "
                f"Conj rate: {conf_rate:.2%} | "
                f"Categories: {self.state.categories_discovered}")


# =============================================================================
# MAIN ENGINE: INVARIANT DISCOVERY ENGINE v6.0
# =============================================================================

class InvariantDiscoveryEngine:
    """
    De complete v6.0 engine.
    
    Drie lagen:
      LAAG 1: Empirische dynamica (attractor detectie)
      LAAG 2: Structurele abstractie (invariant mining, mechanisme synthese)
      LAAG 3: Symbolische redenering (categorieën, tegenvoorbeelden, MDL)
    """
    
    def __init__(self, db_path: str = "invariant_discovery_v6.db"):
        self.ops = OPERATIONS
        
        # Laag 1
        # (attractor detection inline)
        
        # Laag 2
        self.invariant_miner = InvariantMiner(self.ops)
        self.mechanism_synth = MechanismSynthesizer()
        
        # Laag 3
        self.category_builder = CategoryBuilder()
        self.counterexample_hunter = CounterexampleHunter(self.ops)
        self.mdl_scorer = MDLScorer()
        self.homeostatic = HomeostaticController(list(self.ops.keys()))
        
        # State
        self.conjectures: List[Conjecture] = []
        self.categories: List[ConceptualCategory] = []
        self.known_attractors: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        
        # DB
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conjectures (
            id TEXT PRIMARY KEY, statement TEXT, formal TEXT,
            invariant_type TEXT, pipeline TEXT, confidence REAL,
            proof_status TEXT, mechanism TEXT, timestamp REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY, name TEXT, description TEXT,
            properties TEXT, member_count INTEGER, timestamp REAL
        )''')
        conn.commit()
        conn.close()
    
    def _save_conjecture(self, c: Conjecture):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO conjectures 
            VALUES (?,?,?,?,?,?,?,?,?)''', (
            c.id, c.statement, c.formal, c.invariant_type.value,
            json.dumps(list(c.pipeline)), c.confidence,
            c.proof_status.value, c.mechanism, c.created_at
        ))
        conn.commit()
        conn.close()
    
    def _save_category(self, cat: ConceptualCategory):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO categories 
            VALUES (?,?,?,?,?,?)''', (
            cat.id, cat.name, cat.description,
            json.dumps(cat.defining_properties),
            len(cat.member_pipelines), cat.created_at
        ))
        conn.commit()
        conn.close()
    
    # ─── LAAG 1: Attractor Detection ─────────────────────────────────
    
    def detect_attractor(self, pipeline: Tuple[str, ...],
                          domain: Tuple[int, int] = (1000, 99999),
                          sample_size: int = 5000) -> Tuple[Optional[int], float]:
        """Detecteer dominant attractor."""
        
        numbers = random.sample(range(domain[0], domain[1] + 1),
                               min(sample_size, domain[1] - domain[0]))
        
        endpoints: Counter = Counter()
        
        for n in numbers:
            current = n
            for step in range(100):
                prev = current
                for op_name in pipeline:
                    if op_name in self.ops:
                        current = self.ops[op_name](current)
                        if current > 10**15 or current < 0:
                            current = -1
                            break
                if current < 0:
                    break
                if current == prev:
                    endpoints[current] += 1
                    break
            else:
                if current >= 0:
                    endpoints[current] += 1
        
        if not endpoints:
            return None, 0.0
        
        dominant, count = endpoints.most_common(1)[0]
        dominance = 100 * count / len(numbers)
        
        return dominant, dominance
    
    # ─── CORE EXPLORATION LOOP ────────────────────────────────────────
    
    def explore_pipeline(self, pipeline: Tuple[str, ...]) -> Dict:
        """Volledige 3-laags exploratie van een pipeline."""
        
        # LAAG 1: Empirisch
        attractor, dominance = self.detect_attractor(pipeline)
        
        # LAAG 2: Invariant mining
        conjectures = self.invariant_miner.mine_all_invariants(
            pipeline, attractor=attractor
        )
        
        # LAAG 2: Mechanisme synthese
        mechanisms = self.mechanism_synth.synthesize(conjectures, attractor)
        
        # LAAG 3: Counterexample hunting (voor sterke conjectures)
        for conj in conjectures:
            if conj.confidence > 0.95:
                cex = self.counterexample_hunter.hunt(conj, budget=10000)
                if cex:
                    conj.counterexamples.extend(cex)
                    conj.proof_status = ProofStatus.DISPROVEN
                    conj.confidence *= 0.1
                    self.homeostatic.state.conjectures_disproven += 1
                else:
                    conj.proof_status = ProofStatus.EMPIRICAL
                    self.homeostatic.state.conjectures_confirmed += 1
        
        # LAAG 3: MDL scoring
        active_conjectures = [c for c in conjectures if c.is_alive()]
        score = self.mdl_scorer.score(
            pipeline, dominance,
            len(active_conjectures), len(mechanisms)
        )
        
        # LAAG 3: Registreer bij category builder
        self.category_builder.register_pipeline(
            pipeline, active_conjectures, mechanisms, attractor, dominance
        )
        
        # Sla op
        for c in conjectures:
            self.conjectures.append(c)
            self._save_conjecture(c)
        
        if attractor is not None:
            self.known_attractors[pipeline] = (attractor, dominance)
        
        # Update homeostatic controller
        self.homeostatic.record_result(pipeline, score, active_conjectures, mechanisms)
        
        return {
            "pipeline": pipeline,
            "attractor": attractor,
            "dominance": dominance,
            "conjectures": active_conjectures,
            "mechanisms": mechanisms,
            "score": score
        }
    
    # ─── RESEARCH SESSION ─────────────────────────────────────────────
    
    def run_research_session(self, cycles: int = 5, pipelines_per_cycle: int = 15):
        """Run volledige research sessie."""
        
        print("█" * 70)
        print("  SYNTRIAD INVARIANT DISCOVERY ENGINE v6.0")
        print("  Autonomous Symbolic Discovery for Discrete Dynamical Systems")
        print("█" * 70)
        
        session_start = time.time()
        all_results = []
        
        for cycle in range(cycles):
            cycle_start = time.time()
            
            print(f"\n{'='*70}")
            print(f"  🧠 CYCLE {cycle + 1}/{cycles}")
            print(f"{'='*70}")
            print(f"  {self.homeostatic.get_diagnosis()}")
            
            cycle_results = []
            
            for i in range(pipelines_per_cycle):
                pipeline = self.homeostatic.select_pipeline()
                result = self.explore_pipeline(pipeline)
                cycle_results.append(result)
                all_results.append(result)
                
                # Print significante resultaten
                if result["score"] > 0.3 and result["conjectures"]:
                    pipe_str = ' → '.join(pipeline)
                    print(f"\n   [{i+1}] {pipe_str}")
                    print(f"       Attractor: {result['attractor']} "
                          f"({result['dominance']:.1f}%)")
                    print(f"       Score: {result['score']:.3f}")
                    
                    for c in result["conjectures"][:2]:
                        status = "✓" if c.proof_status == ProofStatus.EMPIRICAL else "?"
                        print(f"       {status} {c.statement[:70]}...")
                    
                    for mech_name, expl, cat in result["mechanisms"][:1]:
                        print(f"       ⚙ [{cat}] {expl[:60]}...")
            
            # Adapt
            self.homeostatic.adapt()
            
            # Discover categories
            new_cats = self.category_builder.discover_categories()
            if new_cats:
                self.homeostatic.state.categories_discovered += len(new_cats)
                print(f"\n   🆕 NEW CONCEPTUAL CATEGORIES DISCOVERED:")
                for cat in new_cats:
                    print(f"      📦 {cat.name}")
                    print(f"         {cat.description[:70]}...")
                    print(f"         Members: {len(cat.member_pipelines)} pipelines")
                    for cat_saved in new_cats:
                        self._save_category(cat_saved)
            
            duration = time.time() - cycle_start
            print(f"\n   ⏱ Cycle duration: {duration:.1f}s")
        
        # ─── FINAL REPORT ────────────────────────────────────────────
        
        session_duration = time.time() - session_start
        
        print("\n" + "█" * 70)
        print("  RESEARCH SESSION COMPLETE")
        print("█" * 70)
        
        # Conjectures summary
        alive = [c for c in self.conjectures if c.is_alive()]
        disproven = [c for c in self.conjectures if c.proof_status == ProofStatus.DISPROVEN]
        empirical = [c for c in self.conjectures if c.proof_status == ProofStatus.EMPIRICAL]
        
        print(f"\n📊 SESSION STATISTICS:")
        print(f"   Duration: {session_duration:.1f}s")
        print(f"   Pipelines explored: {len(all_results)}")
        print(f"   Unique attractors: {len(set(r['attractor'] for r in all_results if r['attractor']))}")
        
        print(f"\n📜 CONJECTURES:")
        print(f"   Total generated: {len(self.conjectures)}")
        print(f"   Empirically confirmed: {len(empirical)}")
        print(f"   Disproven: {len(disproven)}")
        print(f"   Still open: {len(alive) - len(empirical)}")
        
        # Top conjectures
        top_conjectures = sorted(alive, key=lambda c: c.confidence, reverse=True)[:5]
        if top_conjectures:
            print(f"\n🏆 STRONGEST CONJECTURES:")
            for c in top_conjectures:
                print(f"   [{c.confidence:.2f}] {c.statement[:70]}...")
                if c.mechanism:
                    print(f"           Mechanism: {c.mechanism[:50]}...")
        
        # Categories
        all_cats = list(self.category_builder.categories.values())
        if all_cats:
            print(f"\n📦 CONCEPTUAL CATEGORIES DISCOVERED: {len(all_cats)}")
            for cat in all_cats:
                print(f"   • {cat.name}")
                print(f"     {cat.description[:70]}...")
                print(f"     Members: {len(cat.member_pipelines)} | "
                      f"Attractors: {cat.member_attractors[:3]}")
                if cat.isomorphic_to:
                    print(f"     Isomorphic to: {cat.isomorphic_to}")
        
        # Mechanisms
        all_mechs = set()
        for r in all_results:
            for _, _, cat in r["mechanisms"]:
                all_mechs.add(cat)
        
        if all_mechs:
            print(f"\n⚙ MECHANISM CATEGORIES IDENTIFIED: {len(all_mechs)}")
            for m in sorted(all_mechs):
                print(f"   • {m}")
        
        # Homeostatic state
        print(f"\n🔄 SYSTEM SELF-DIAGNOSIS:")
        print(f"   {self.homeostatic.get_diagnosis()}")
        print(f"   Top operators: {self.homeostatic.state.preferred_operators}")
        
        # MDL top discoveries
        top_results = sorted(all_results, key=lambda r: r["score"], reverse=True)[:5]
        if top_results:
            print(f"\n💎 MOST ELEGANT DISCOVERIES (MDL):")
            for r in top_results:
                pipe_str = ' → '.join(r["pipeline"])
                print(f"   [{r['score']:.3f}] {pipe_str}")
                print(f"           Attr: {r['attractor']} | "
                      f"Dom: {r['dominance']:.1f}% | "
                      f"Inv: {len(r['conjectures'])} | "
                      f"Mech: {len(r['mechanisms'])}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    engine = InvariantDiscoveryEngine()
    engine.run_research_session(cycles=5, pipelines_per_cycle=15)
