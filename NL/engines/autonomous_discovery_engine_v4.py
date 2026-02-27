#!/usr/bin/env python3
"""
SYNTRIAD Autonomous Discovery Engine v4.0
==========================================

Een echte autonome onderzoeker die:
1. Zelf nieuwe pipeline-combinaties genereert en test
2. Basin-of-attraction structuren analyseert
3. Uitzonderingen identificeert en classificeert
4. Hypotheses formuleert op basis van patronen
5. Algebraïsche reducties detecteert
6. Zichzelf iteratief verbetert

Dit is geen brute-force explorer meer, maar een discovery engine
die patronen herkent en nieuwe hypotheses genereert.

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
import json
import sqlite3
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable
from collections import Counter, defaultdict
from pathlib import Path
from enum import Enum
import itertools

try:
    from numba import cuda, int64, njit, prange
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


# =============================================================================
# DIGIT OPERATIONS LIBRARY
# =============================================================================

class DigitOp:
    """Base class voor digit operaties."""
    
    @staticmethod
    def reverse(n: int) -> int:
        return int(str(abs(n))[::-1]) if n != 0 else 0
    
    @staticmethod
    def digit_sum(n: int) -> int:
        return sum(int(d) for d in str(abs(n)))
    
    @staticmethod
    def digit_product(n: int) -> int:
        result = 1
        for d in str(abs(n)):
            if int(d) > 0:
                result *= int(d)
        return result
    
    @staticmethod
    def digit_pow(n: int, power: int) -> int:
        return sum(int(d)**power for d in str(abs(n)))
    
    @staticmethod
    def digit_pow2(n: int) -> int:
        return DigitOp.digit_pow(n, 2)
    
    @staticmethod
    def digit_pow3(n: int) -> int:
        return DigitOp.digit_pow(n, 3)
    
    @staticmethod
    def digit_pow4(n: int) -> int:
        return DigitOp.digit_pow(n, 4)
    
    @staticmethod
    def digit_pow5(n: int) -> int:
        return DigitOp.digit_pow(n, 5)
    
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
        result = s[-1] + s[1:-1] + s[0]
        return int(result.lstrip('0') or '0')
    
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
    def happy_step(n: int) -> int:
        return DigitOp.digit_pow2(n)
    
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


# Operation registry
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
    'happy_step': DigitOp.happy_step,
    'rotate_left': DigitOp.rotate_left,
    'rotate_right': DigitOp.rotate_right,
}


# =============================================================================
# HYPOTHESIS SYSTEM
# =============================================================================

class HypothesisType(Enum):
    FIXED_POINT = "fixed_point"
    CYCLE = "cycle"
    DOMINANT_ATTRACTOR = "dominant_attractor"
    MULTI_ATTRACTOR = "multi_attractor"
    DIGIT_DEPENDENT = "digit_dependent"
    ALGEBRAIC_RELATION = "algebraic_relation"


@dataclass
class Hypothesis:
    """Een wiskundige hypothese gegenereerd door het systeem."""
    id: str
    type: HypothesisType
    pipeline: Tuple[str, ...]
    claim: str
    evidence: Dict
    confidence: float  # 0-1
    status: str  # "generated", "testing", "confirmed", "refuted", "refined"
    refinements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.value,
            'pipeline': list(self.pipeline),
            'claim': self.claim,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'status': self.status,
            'refinements': self.refinements
        }


# =============================================================================
# BASIN OF ATTRACTION ANALYZER
# =============================================================================

@dataclass
class BasinInfo:
    """Informatie over een basin of attraction."""
    attractor: Tuple[int, ...]  # Fixed point of cycle
    basin_size: int
    basin_percentage: float
    avg_steps_to_attractor: float
    max_steps: int
    example_trajectories: List[List[int]]
    boundary_numbers: List[int]  # Numbers at basin boundary


@dataclass 
class AttractorGraph:
    """Complete attractor structuur van een pipeline."""
    pipeline: Tuple[str, ...]
    domain_tested: Tuple[int, int]
    total_tested: int
    basins: List[BasinInfo]
    is_monostable: bool
    dominant_attractor: Optional[Tuple[int, ...]]
    dominance_ratio: float
    exceptions: List[int]
    exception_analysis: Dict


class BasinAnalyzer:
    """Analyseert basin-of-attraction structuren."""
    
    def __init__(self, operations: Dict[str, Callable]):
        self.ops = operations
    
    def apply_pipeline(self, n: int, pipeline: Tuple[str, ...], max_iter: int = 500) -> Tuple[int, List[int]]:
        """Pas pipeline toe en return (eindpunt, traject)."""
        trajectory = [n]
        seen = {n: 0}
        current = n
        
        for step in range(max_iter):
            for op_name in pipeline:
                if op_name in self.ops:
                    current = self.ops[op_name](current)
                    if current > 10**15:  # Overflow protection
                        return -1, trajectory
            
            if current in seen:
                trajectory.append(current)
                return current, trajectory
            
            seen[current] = len(trajectory)
            trajectory.append(current)
        
        return current, trajectory
    
    def detect_cycle(self, n: int, pipeline: Tuple[str, ...], max_iter: int = 500) -> Tuple[Tuple[int, ...], int]:
        """Detecteer cyclus en return (cycle_elements, steps_to_cycle)."""
        trajectory = [n]
        seen = {n: 0}
        current = n
        
        for step in range(1, max_iter + 1):
            for op_name in pipeline:
                if op_name in self.ops:
                    current = self.ops[op_name](current)
            
            if current in seen:
                cycle_start = seen[current]
                cycle = tuple(trajectory[cycle_start:])
                return cycle, cycle_start
            
            seen[current] = step
            trajectory.append(current)
        
        return (current,), max_iter
    
    def analyze_basins(self, pipeline: Tuple[str, ...], 
                       domain: Tuple[int, int],
                       sample_size: Optional[int] = None) -> AttractorGraph:
        """Volledige basin-of-attraction analyse."""
        
        low, high = domain
        total = high - low + 1
        
        if sample_size and sample_size < total:
            numbers = random.sample(range(low, high + 1), sample_size)
        else:
            numbers = range(low, high + 1)
            sample_size = total
        
        # Collect all cycles/attractors
        attractor_to_numbers: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        attractor_to_steps: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        attractor_to_trajectories: Dict[Tuple[int, ...], List[List[int]]] = defaultdict(list)
        
        for n in numbers:
            cycle, steps = self.detect_cycle(n, pipeline)
            attractor_to_numbers[cycle].append(n)
            attractor_to_steps[cycle].append(steps)
            
            if len(attractor_to_trajectories[cycle]) < 5:
                _, traj = self.apply_pipeline(n, pipeline)
                attractor_to_trajectories[cycle].append(traj[:20])  # First 20 steps
        
        # Build basin info
        basins = []
        for attractor, numbers_list in attractor_to_numbers.items():
            basin = BasinInfo(
                attractor=attractor,
                basin_size=len(numbers_list),
                basin_percentage=100 * len(numbers_list) / sample_size,
                avg_steps_to_attractor=np.mean(attractor_to_steps[attractor]),
                max_steps=max(attractor_to_steps[attractor]),
                example_trajectories=attractor_to_trajectories[attractor],
                boundary_numbers=[]  # TODO: identify boundary
            )
            basins.append(basin)
        
        # Sort by size
        basins.sort(key=lambda b: b.basin_size, reverse=True)
        
        # Determine dominance
        is_monostable = len(basins) == 1
        dominant = basins[0].attractor if basins else None
        dominance = basins[0].basin_percentage if basins else 0
        
        # Collect exceptions (non-dominant)
        exceptions = []
        for basin in basins[1:]:
            exceptions.extend(attractor_to_numbers[basin.attractor][:10])
        
        # Analyze exceptions
        exception_analysis = self._analyze_exceptions(exceptions, pipeline)
        
        return AttractorGraph(
            pipeline=pipeline,
            domain_tested=domain,
            total_tested=sample_size,
            basins=basins,
            is_monostable=is_monostable,
            dominant_attractor=dominant,
            dominance_ratio=dominance,
            exceptions=exceptions[:50],
            exception_analysis=exception_analysis
        )
    
    def _analyze_exceptions(self, exceptions: List[int], pipeline: Tuple[str, ...]) -> Dict:
        """Analyseer waarom uitzonderingen niet naar dominant attractor gaan."""
        if not exceptions:
            return {"count": 0, "patterns": []}
        
        patterns = {
            "palindromes": 0,
            "repdigits": 0,
            "symmetric": 0,
            "leading_zero_collapse": 0,
            "small_numbers": 0,
            "other": 0
        }
        
        for n in exceptions:
            s = str(n)
            if s == s[::-1]:
                patterns["palindromes"] += 1
            elif len(set(s)) == 1:
                patterns["repdigits"] += 1
            elif n < 100:
                patterns["small_numbers"] += 1
            else:
                patterns["other"] += 1
        
        return {
            "count": len(exceptions),
            "patterns": patterns,
            "examples": exceptions[:10]
        }


# =============================================================================
# ALGEBRAIC PATTERN DETECTOR
# =============================================================================

class AlgebraicDetector:
    """Detecteert algebraïsche patronen en relaties."""
    
    @staticmethod
    def factorize(n: int) -> Dict[int, int]:
        """Priemfactorisatie."""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    @staticmethod
    def is_perfect_power(n: int) -> Optional[Tuple[int, int]]:
        """Check of n = a^b voor b > 1."""
        if n <= 1:
            return None
        for b in range(2, int(math.log2(n)) + 2):
            a = round(n ** (1/b))
            for candidate in [a-1, a, a+1]:
                if candidate > 0 and candidate ** b == n:
                    return (candidate, b)
        return None
    
    @staticmethod
    def digit_structure(n: int) -> Dict:
        """Analyseer digit structuur."""
        s = str(n)
        digits = [int(d) for d in s]
        return {
            "length": len(s),
            "digit_sum": sum(digits),
            "digit_product": math.prod(d for d in digits if d > 0),
            "is_palindrome": s == s[::-1],
            "is_repdigit": len(set(s)) == 1,
            "unique_digits": len(set(s)),
            "has_zero": '0' in s,
            "digit_mean": sum(digits) / len(digits),
            "digit_variance": np.var(digits) if len(digits) > 1 else 0
        }
    
    @staticmethod
    def find_algebraic_relation(a: int, b: int) -> Optional[str]:
        """Zoek algebraïsche relatie tussen twee getallen."""
        relations = []
        
        # Check simple relations
        if a + b == 10**len(str(a)):
            relations.append(f"{a} + {b} = 10^{len(str(a))}")
        
        if a * b == 10**len(str(a)):
            relations.append(f"{a} × {b} = 10^{len(str(a))}")
        
        # Check power relations
        power_a = AlgebraicDetector.is_perfect_power(a)
        power_b = AlgebraicDetector.is_perfect_power(b)
        
        if power_a:
            relations.append(f"{a} = {power_a[0]}^{power_a[1]}")
        if power_b:
            relations.append(f"{b} = {power_b[0]}^{power_b[1]}")
        
        # Check if they share factors
        factors_a = AlgebraicDetector.factorize(a)
        factors_b = AlgebraicDetector.factorize(b)
        
        common = set(factors_a.keys()) & set(factors_b.keys())
        if common:
            relations.append(f"Common factors: {common}")
        
        return "; ".join(relations) if relations else None


# =============================================================================
# HYPOTHESIS GENERATOR
# =============================================================================

class HypothesisGenerator:
    """Genereert hypotheses op basis van observaties."""
    
    def __init__(self):
        self.hypothesis_counter = 0
    
    def generate_id(self) -> str:
        self.hypothesis_counter += 1
        return f"H{self.hypothesis_counter:04d}"
    
    def from_basin_analysis(self, graph: AttractorGraph) -> List[Hypothesis]:
        """Genereer hypotheses uit basin analyse."""
        hypotheses = []
        
        # Hypothesis 1: Dominant attractor
        if graph.dominance_ratio > 95:
            h = Hypothesis(
                id=self.generate_id(),
                type=HypothesisType.DOMINANT_ATTRACTOR,
                pipeline=graph.pipeline,
                claim=f"Pipeline converges to {graph.dominant_attractor} for {graph.dominance_ratio:.2f}% of inputs in domain {graph.domain_tested}",
                evidence={
                    "dominance_ratio": graph.dominance_ratio,
                    "total_tested": graph.total_tested,
                    "exception_count": len(graph.exceptions)
                },
                confidence=min(0.99, graph.dominance_ratio / 100),
                status="generated"
            )
            hypotheses.append(h)
        
        # Hypothesis 2: Multi-attractor system
        if len(graph.basins) > 1:
            h = Hypothesis(
                id=self.generate_id(),
                type=HypothesisType.MULTI_ATTRACTOR,
                pipeline=graph.pipeline,
                claim=f"Pipeline has {len(graph.basins)} distinct attractors",
                evidence={
                    "attractors": [b.attractor for b in graph.basins[:5]],
                    "basin_sizes": [b.basin_percentage for b in graph.basins[:5]]
                },
                confidence=0.95,
                status="generated"
            )
            hypotheses.append(h)
        
        # Hypothesis 3: Exception pattern
        if graph.exception_analysis["count"] > 0:
            patterns = graph.exception_analysis["patterns"]
            dominant_pattern = max(patterns.items(), key=lambda x: x[1])
            
            if dominant_pattern[1] > 0:
                h = Hypothesis(
                    id=self.generate_id(),
                    type=HypothesisType.DIGIT_DEPENDENT,
                    pipeline=graph.pipeline,
                    claim=f"Exceptions are primarily {dominant_pattern[0]} ({dominant_pattern[1]} cases)",
                    evidence=graph.exception_analysis,
                    confidence=0.8,
                    status="generated"
                )
                hypotheses.append(h)
        
        return hypotheses
    
    def from_algebraic_analysis(self, attractor: int, pipeline: Tuple[str, ...]) -> List[Hypothesis]:
        """Genereer hypotheses uit algebraïsche analyse."""
        hypotheses = []
        
        # Check perfect power
        power = AlgebraicDetector.is_perfect_power(attractor)
        if power:
            h = Hypothesis(
                id=self.generate_id(),
                type=HypothesisType.ALGEBRAIC_RELATION,
                pipeline=pipeline,
                claim=f"Attractor {attractor} = {power[0]}^{power[1]} suggests power-sum stability",
                evidence={
                    "base": power[0],
                    "exponent": power[1],
                    "factorization": AlgebraicDetector.factorize(attractor)
                },
                confidence=0.7,
                status="generated"
            )
            hypotheses.append(h)
        
        # Check digit structure
        structure = AlgebraicDetector.digit_structure(attractor)
        if structure["is_palindrome"]:
            h = Hypothesis(
                id=self.generate_id(),
                type=HypothesisType.ALGEBRAIC_RELATION,
                pipeline=pipeline,
                claim=f"Attractor {attractor} is palindromic - may indicate symmetric stability",
                evidence=structure,
                confidence=0.6,
                status="generated"
            )
            hypotheses.append(h)
        
        return hypotheses


# =============================================================================
# PIPELINE GENERATOR
# =============================================================================

class PipelineGenerator:
    """Genereert nieuwe pipeline combinaties intelligent."""
    
    def __init__(self, operations: List[str]):
        self.ops = operations
        self.tested_pipelines: Set[Tuple[str, ...]] = set()
        self.successful_patterns: List[Tuple[str, ...]] = []
        
        # Operation categories for smart generation
        self.categories = {
            "reducing": ["digit_sum", "digit_pow2", "digit_pow3", "digit_pow4"],
            "transforming": ["reverse", "sort_asc", "sort_desc", "swap_ends", "rotate_left"],
            "combining": ["kaprekar_step", "truc_1089", "add_reverse", "sub_reverse"],
            "special": ["digit_factorial_sum", "happy_step", "complement_9"]
        }
    
    def generate_random(self, length: int = 2) -> Tuple[str, ...]:
        """Genereer random pipeline."""
        pipeline = tuple(random.choices(self.ops, k=length))
        self.tested_pipelines.add(pipeline)
        return pipeline
    
    def generate_structured(self, pattern: str = "reduce_transform") -> Tuple[str, ...]:
        """Genereer pipeline met structuur."""
        if pattern == "reduce_transform":
            # Reducing op followed by transforming op
            return (
                random.choice(self.categories["reducing"]),
                random.choice(self.categories["transforming"])
            )
        elif pattern == "transform_reduce":
            return (
                random.choice(self.categories["transforming"]),
                random.choice(self.categories["reducing"])
            )
        elif pattern == "combine_reduce":
            return (
                random.choice(self.categories["combining"]),
                random.choice(self.categories["reducing"])
            )
        elif pattern == "double_combine":
            return (
                random.choice(self.categories["combining"]),
                random.choice(self.categories["combining"])
            )
        else:
            return self.generate_random(2)
    
    def generate_mutation(self, pipeline: Tuple[str, ...]) -> Tuple[str, ...]:
        """Muteer een bestaande pipeline."""
        pipeline = list(pipeline)
        mutation_type = random.choice(["replace", "insert", "delete", "swap"])
        
        if mutation_type == "replace" and pipeline:
            idx = random.randrange(len(pipeline))
            pipeline[idx] = random.choice(self.ops)
        elif mutation_type == "insert" and len(pipeline) < 5:
            idx = random.randrange(len(pipeline) + 1)
            pipeline.insert(idx, random.choice(self.ops))
        elif mutation_type == "delete" and len(pipeline) > 1:
            idx = random.randrange(len(pipeline))
            pipeline.pop(idx)
        elif mutation_type == "swap" and len(pipeline) >= 2:
            i, j = random.sample(range(len(pipeline)), 2)
            pipeline[i], pipeline[j] = pipeline[j], pipeline[i]
        
        return tuple(pipeline)
    
    def generate_crossover(self, p1: Tuple[str, ...], p2: Tuple[str, ...]) -> Tuple[str, ...]:
        """Crossover van twee pipelines."""
        if len(p1) == 0 or len(p2) == 0:
            return p1 or p2
        
        cut1 = random.randrange(len(p1))
        cut2 = random.randrange(len(p2))
        
        child = p1[:cut1] + p2[cut2:]
        return child if child else p1
    
    def record_success(self, pipeline: Tuple[str, ...]):
        """Registreer succesvolle pipeline."""
        self.successful_patterns.append(pipeline)
    
    def generate_from_success(self) -> Optional[Tuple[str, ...]]:
        """Genereer nieuwe pipeline gebaseerd op successen."""
        if not self.successful_patterns:
            return None
        
        base = random.choice(self.successful_patterns)
        return self.generate_mutation(base)


# =============================================================================
# AUTONOMOUS DISCOVERY ENGINE
# =============================================================================

class AutonomousDiscoveryEngine:
    """De autonome onderzoeksengine."""
    
    def __init__(self, db_path: str = "autonomous_discoveries_v4.db"):
        self.ops = OPERATIONS
        self.basin_analyzer = BasinAnalyzer(self.ops)
        self.hypothesis_generator = HypothesisGenerator()
        self.pipeline_generator = PipelineGenerator(list(self.ops.keys()))
        self.algebraic_detector = AlgebraicDetector()
        
        self.hypotheses: List[Hypothesis] = []
        self.discoveries: List[Dict] = []
        self.iteration = 0
        
        # Database
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialiseer database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            pipeline TEXT,
            attractor TEXT,
            dominance REAL,
            hypothesis_type TEXT,
            claim TEXT,
            evidence TEXT,
            confidence REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS hypotheses (
            id TEXT PRIMARY KEY,
            type TEXT,
            pipeline TEXT,
            claim TEXT,
            evidence TEXT,
            confidence REAL,
            status TEXT
        )''')
        conn.commit()
        conn.close()
    
    def _save_hypothesis(self, h: Hypothesis):
        """Sla hypothese op in database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO hypotheses VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (h.id, h.type.value, str(h.pipeline), h.claim, 
                   json.dumps(h.evidence), h.confidence, h.status))
        conn.commit()
        conn.close()
    
    def explore_pipeline(self, pipeline: Tuple[str, ...], 
                         domain: Tuple[int, int] = (1000, 99999),
                         sample_size: int = 10000) -> AttractorGraph:
        """Exploreer een pipeline volledig."""
        return self.basin_analyzer.analyze_basins(pipeline, domain, sample_size)
    
    def generate_hypotheses(self, graph: AttractorGraph) -> List[Hypothesis]:
        """Genereer hypotheses uit analyse."""
        hypotheses = []
        
        # Basin-based hypotheses
        hypotheses.extend(self.hypothesis_generator.from_basin_analysis(graph))
        
        # Algebraic hypotheses for dominant attractor
        if graph.dominant_attractor and len(graph.dominant_attractor) == 1:
            attractor = graph.dominant_attractor[0]
            hypotheses.extend(
                self.hypothesis_generator.from_algebraic_analysis(attractor, graph.pipeline)
            )
        
        return hypotheses
    
    def test_hypothesis(self, h: Hypothesis, extended_domain: Tuple[int, int]) -> Hypothesis:
        """Test een hypothese op uitgebreid domein."""
        h.status = "testing"
        
        # Re-analyze on extended domain
        graph = self.explore_pipeline(h.pipeline, extended_domain, sample_size=50000)
        
        # Update confidence based on new evidence
        if h.type == HypothesisType.DOMINANT_ATTRACTOR:
            new_dominance = graph.dominance_ratio
            old_dominance = h.evidence.get("dominance_ratio", 0)
            
            if abs(new_dominance - old_dominance) < 2:
                h.confidence = min(0.99, h.confidence + 0.1)
                h.status = "confirmed"
                h.refinements.append(f"Extended test: {new_dominance:.2f}% on {extended_domain}")
            else:
                h.confidence = max(0.1, h.confidence - 0.2)
                h.status = "refined"
                h.refinements.append(f"Dominance changed: {old_dominance:.2f}% → {new_dominance:.2f}%")
        
        return h
    
    def run_discovery_cycle(self, num_pipelines: int = 10, 
                            domain: Tuple[int, int] = (1000, 99999)) -> List[Hypothesis]:
        """Run één discovery cyclus."""
        self.iteration += 1
        print(f"\n{'='*70}")
        print(f"🔬 DISCOVERY CYCLE {self.iteration}")
        print(f"{'='*70}")
        
        cycle_hypotheses = []
        
        for i in range(num_pipelines):
            # Generate pipeline
            if random.random() < 0.3 and self.pipeline_generator.successful_patterns:
                pipeline = self.pipeline_generator.generate_from_success()
            elif random.random() < 0.5:
                pattern = random.choice(["reduce_transform", "transform_reduce", 
                                         "combine_reduce", "double_combine"])
                pipeline = self.pipeline_generator.generate_structured(pattern)
            else:
                pipeline = self.pipeline_generator.generate_random(random.randint(2, 4))
            
            if not pipeline:
                continue
            
            print(f"\n   [{i+1}/{num_pipelines}] Testing: {' → '.join(pipeline)}")
            
            # Analyze
            try:
                graph = self.explore_pipeline(pipeline, domain, sample_size=5000)
            except Exception as e:
                print(f"      ⚠️ Error: {e}")
                continue
            
            # Report
            if graph.dominant_attractor:
                print(f"      Dominant: {graph.dominant_attractor} ({graph.dominance_ratio:.1f}%)")
                print(f"      Basins: {len(graph.basins)}, Exceptions: {len(graph.exceptions)}")
                
                # Generate hypotheses
                hypotheses = self.generate_hypotheses(graph)
                
                for h in hypotheses:
                    print(f"      📝 {h.type.value}: {h.claim[:60]}...")
                    self._save_hypothesis(h)
                    cycle_hypotheses.append(h)
                
                # Record success if dominant
                if graph.dominance_ratio > 90:
                    self.pipeline_generator.record_success(pipeline)
        
        self.hypotheses.extend(cycle_hypotheses)
        return cycle_hypotheses
    
    def analyze_exceptions_deeply(self, pipeline: Tuple[str, ...], 
                                   domain: Tuple[int, int] = (1000, 99999)) -> Dict:
        """Diepgaande analyse van uitzonderingen."""
        print(f"\n{'='*70}")
        print(f"🔍 EXCEPTION ANALYSIS: {' → '.join(pipeline)}")
        print(f"{'='*70}")
        
        graph = self.explore_pipeline(pipeline, domain, sample_size=50000)
        
        if not graph.exceptions:
            print("   No exceptions found!")
            return {"exceptions": [], "patterns": {}}
        
        print(f"\n   Found {len(graph.exceptions)} exceptions")
        print(f"   Exception patterns: {graph.exception_analysis['patterns']}")
        
        # Detailed analysis of each exception
        exception_details = []
        for n in graph.exceptions[:20]:
            cycle, steps = self.basin_analyzer.detect_cycle(n, pipeline)
            structure = self.algebraic_detector.digit_structure(n)
            
            detail = {
                "number": n,
                "converges_to": cycle,
                "steps": steps,
                "structure": structure
            }
            exception_details.append(detail)
            
            print(f"\n   {n}:")
            print(f"      → Converges to: {cycle}")
            print(f"      → Steps: {steps}")
            print(f"      → Palindrome: {structure['is_palindrome']}, "
                  f"Repdigit: {structure['is_repdigit']}")
        
        return {
            "exceptions": exception_details,
            "patterns": graph.exception_analysis["patterns"],
            "dominant_attractor": graph.dominant_attractor,
            "dominance": graph.dominance_ratio
        }
    
    def find_algebraic_fixed_points(self, pipeline: Tuple[str, ...], 
                                     search_range: Tuple[int, int] = (1, 1000000)) -> List[int]:
        """Vind algebraïsche fixed points."""
        print(f"\n{'='*70}")
        print(f"🧮 ALGEBRAIC FIXED POINT SEARCH: {' → '.join(pipeline)}")
        print(f"{'='*70}")
        
        fixed_points = []
        
        for n in range(search_range[0], min(search_range[1], 100000)):
            # Apply pipeline once
            result = n
            for op_name in pipeline:
                if op_name in self.ops:
                    result = self.ops[op_name](result)
            
            if result == n:
                fixed_points.append(n)
                power = self.algebraic_detector.is_perfect_power(n)
                factors = self.algebraic_detector.factorize(n)
                print(f"   Fixed point: {n}")
                if power:
                    print(f"      = {power[0]}^{power[1]}")
                print(f"      Factors: {factors}")
        
        print(f"\n   Total fixed points found: {len(fixed_points)}")
        return fixed_points
    
    def run_full_research_session(self, cycles: int = 5):
        """Run volledige onderzoekssessie."""
        print("█" * 70)
        print("  SYNTRIAD AUTONOMOUS DISCOVERY ENGINE v4.0")
        print("  Fully Autonomous Mathematical Research System")
        print("█" * 70)
        
        start_time = time.time()
        
        # Phase 1: Exploration
        print("\n" + "▓" * 70)
        print("  PHASE 1: EXPLORATION")
        print("▓" * 70)
        
        for cycle in range(cycles):
            self.run_discovery_cycle(num_pipelines=10)
        
        # Phase 2: Test top hypotheses
        print("\n" + "▓" * 70)
        print("  PHASE 2: HYPOTHESIS TESTING")
        print("▓" * 70)
        
        # Sort by confidence
        top_hypotheses = sorted(
            [h for h in self.hypotheses if h.type == HypothesisType.DOMINANT_ATTRACTOR],
            key=lambda h: h.confidence,
            reverse=True
        )[:5]
        
        for h in top_hypotheses:
            print(f"\n   Testing: {h.claim[:60]}...")
            h = self.test_hypothesis(h, (1000, 999999))
            print(f"   Result: {h.status} (confidence: {h.confidence:.2f})")
        
        # Phase 3: Exception analysis for top findings
        print("\n" + "▓" * 70)
        print("  PHASE 3: EXCEPTION ANALYSIS")
        print("▓" * 70)
        
        for h in top_hypotheses[:3]:
            if h.status == "confirmed":
                self.analyze_exceptions_deeply(h.pipeline)
        
        # Phase 4: Algebraic analysis
        print("\n" + "▓" * 70)
        print("  PHASE 4: ALGEBRAIC ANALYSIS")
        print("▓" * 70)
        
        for h in top_hypotheses[:2]:
            self.find_algebraic_fixed_points(h.pipeline)
        
        # Summary
        elapsed = time.time() - start_time
        
        print("\n" + "█" * 70)
        print("  RESEARCH SESSION COMPLETE")
        print("█" * 70)
        print(f"\n   Duration: {elapsed:.1f}s")
        print(f"   Iterations: {self.iteration}")
        print(f"   Hypotheses generated: {len(self.hypotheses)}")
        print(f"   Confirmed: {sum(1 for h in self.hypotheses if h.status == 'confirmed')}")
        print(f"   Refuted: {sum(1 for h in self.hypotheses if h.status == 'refuted')}")
        
        # Top discoveries
        print("\n   📋 TOP DISCOVERIES:")
        for h in sorted(self.hypotheses, key=lambda h: h.confidence, reverse=True)[:5]:
            print(f"      [{h.confidence:.2f}] {h.claim[:70]}...")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    engine = AutonomousDiscoveryEngine()
    engine.run_full_research_session(cycles=3)
