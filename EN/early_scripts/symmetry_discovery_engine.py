#!/usr/bin/env python3
"""
SYNTRIAD Symmetry Discovery Engine v1.0
=======================================

An AI-driven system for autonomous discovery of symmetries in numbers.

Core concepts:
- Operation: A transformation on numbers (reverse, digitsum, sort, etc.)
- Pipeline: A composition of operations
- Attractor: A fixed point or cycle where a pipeline converges to
- Symmetry: A pattern that holds for a class of numbers

Hardware: Optimized for RTX 4000 Ada, 32-core i9, 64GB RAM
Author: SYNTRIAD Research
"""

import numpy as np
import json
import time
import hashlib
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from pathlib import Path
import random
import math
from abc import ABC, abstractmethod
from enum import Enum, auto
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DiscoveryConfig:
    """Configuration for the discovery engine."""
    num_cpu_workers: int = 30
    gpu_batch_size: int = 1_000_000
    max_pipeline_length: int = 5
    min_test_numbers: int = 500
    max_test_numbers: int = 5_000
    max_iterations_per_pipeline: int = 100
    population_size: int = 100
    elite_fraction: float = 0.1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.4
    novelty_weight: float = 0.3
    thompson_alpha: float = 1.0
    db_path: str = "discoveries.db"
    log_interval: int = 1
    digit_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (3, 3), (4, 4), (5, 5),
    ])


# =============================================================================
# OPERATIONS - The building blocks of transformations
# =============================================================================

class Operation(ABC):
    """Abstract base class for number operations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def apply(self, n: int) -> int:
        pass
    
    def apply_batch(self, arr: np.ndarray) -> np.ndarray:
        return np.array([self.apply(int(x)) for x in arr], dtype=np.int64)
    
    def __repr__(self):
        return self.name


class ReverseOp(Operation):
    @property
    def name(self) -> str:
        return "reverse"
    
    def apply(self, n: int) -> int:
        return int(str(abs(n))[::-1]) * (1 if n >= 0 else -1)


class DigitSumOp(Operation):
    @property
    def name(self) -> str:
        return "digitsum"
    
    def apply(self, n: int) -> int:
        return sum(int(d) for d in str(abs(n)))


class DigitProductOp(Operation):
    @property
    def name(self) -> str:
        return "digitprod"
    
    def apply(self, n: int) -> int:
        result = 1
        for d in str(abs(n)):
            result *= int(d)
        return result


class SortDescOp(Operation):
    @property
    def name(self) -> str:
        return "sort_desc"
    
    def apply(self, n: int) -> int:
        digits = sorted(str(abs(n)), reverse=True)
        return int(''.join(digits)) if digits else 0


class SortAscOp(Operation):
    @property
    def name(self) -> str:
        return "sort_asc"
    
    def apply(self, n: int) -> int:
        digits = sorted(str(abs(n)))
        result = ''.join(digits).lstrip('0') or '0'
        return int(result)


class KaprekarStepOp(Operation):
    @property
    def name(self) -> str:
        return "kaprekar_step"
    
    def apply(self, n: int) -> int:
        s = str(abs(n))
        groot = int(''.join(sorted(s, reverse=True)))
        klein_digits = ''.join(sorted(s)).lstrip('0') or '0'
        klein = int(klein_digits)
        return groot - klein


class Truc1089Op(Operation):
    @property
    def name(self) -> str:
        return "truc_1089"
    
    def apply(self, n: int) -> int:
        n = abs(n)
        rev = int(str(n)[::-1])
        diff = abs(n - rev)
        if diff == 0:
            return 0
        rev_diff = int(str(diff)[::-1])
        return diff + rev_diff


class AddReverseOp(Operation):
    @property
    def name(self) -> str:
        return "add_reverse"
    
    def apply(self, n: int) -> int:
        n = abs(n)
        return n + int(str(n)[::-1])


class SubReverseOp(Operation):
    @property
    def name(self) -> str:
        return "sub_reverse"
    
    def apply(self, n: int) -> int:
        n = abs(n)
        return abs(n - int(str(n)[::-1]))


class MulReverseOp(Operation):
    @property
    def name(self) -> str:
        return "mul_reverse"
    
    def apply(self, n: int) -> int:
        n = abs(n)
        return n * int(str(n)[::-1])


class XorReverseOp(Operation):
    @property
    def name(self) -> str:
        return "xor_reverse"
    
    def apply(self, n: int) -> int:
        n = abs(n)
        return n ^ int(str(n)[::-1])


class SquareOp(Operation):
    @property
    def name(self) -> str:
        return "square"
    
    def apply(self, n: int) -> int:
        return n * n


class CollatzStepOp(Operation):
    @property
    def name(self) -> str:
        return "collatz_step"
    
    def apply(self, n: int) -> int:
        n = abs(n)
        if n == 0:
            return 0
        return n // 2 if n % 2 == 0 else 3 * n + 1


class DigitRotateLeftOp(Operation):
    @property
    def name(self) -> str:
        return "rotate_left"
    
    def apply(self, n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        rotated = s[1:] + s[0]
        return int(rotated.lstrip('0') or '0')


class DigitRotateRightOp(Operation):
    @property
    def name(self) -> str:
        return "rotate_right"
    
    def apply(self, n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        rotated = s[-1] + s[:-1]
        return int(rotated.lstrip('0') or '0')


class DigitMirrorOp(Operation):
    @property
    def name(self) -> str:
        return "mirror"
    
    def apply(self, n: int) -> int:
        s = str(abs(n))
        return int(s + s[::-1])


class DigitAlternatingSum(Operation):
    @property
    def name(self) -> str:
        return "alt_sum"
    
    def apply(self, n: int) -> int:
        result = 0
        sign = 1
        for d in str(abs(n)):
            result += sign * int(d)
            sign *= -1
        return abs(result)


class DigitMaxMinDiff(Operation):
    @property
    def name(self) -> str:
        return "maxmin_diff"
    
    def apply(self, n: int) -> int:
        digits = [int(d) for d in str(abs(n))]
        return max(digits) - min(digits)


class HappyNumberStepOp(Operation):
    @property
    def name(self) -> str:
        return "happy_step"
    
    def apply(self, n: int) -> int:
        return sum(int(d)**2 for d in str(abs(n)))


class DigitFactorialSumOp(Operation):
    FACTORIALS = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
    
    @property
    def name(self) -> str:
        return "digit_factorial_sum"
    
    def apply(self, n: int) -> int:
        return sum(self.FACTORIALS[int(d)] for d in str(abs(n)))


class DigitPairSumOp(Operation):
    @property
    def name(self) -> str:
        return "pair_sum"
    
    def apply(self, n: int) -> int:
        digits = [int(d) for d in str(abs(n))]
        if len(digits) < 2:
            return n
        pairs = [digits[i] + digits[i+1] for i in range(len(digits) - 1)]
        return int(''.join(str(p) for p in pairs)) if pairs else 0


# Registry of all operations
OPERATION_REGISTRY: Dict[str, Operation] = {
    'reverse': ReverseOp(),
    'digitsum': DigitSumOp(),
    'digitprod': DigitProductOp(),
    'sort_desc': SortDescOp(),
    'sort_asc': SortAscOp(),
    'kaprekar_step': KaprekarStepOp(),
    'truc_1089': Truc1089Op(),
    'add_reverse': AddReverseOp(),
    'sub_reverse': SubReverseOp(),
    'mul_reverse': MulReverseOp(),
    'xor_reverse': XorReverseOp(),
    'square': SquareOp(),
    'collatz_step': CollatzStepOp(),
    'rotate_left': DigitRotateLeftOp(),
    'rotate_right': DigitRotateRightOp(),
    'mirror': DigitMirrorOp(),
    'alt_sum': DigitAlternatingSum(),
    'maxmin_diff': DigitMaxMinDiff(),
    'happy_step': HappyNumberStepOp(),
    'digit_factorial_sum': DigitFactorialSumOp(),
    'pair_sum': DigitPairSumOp(),
}


# =============================================================================
# PIPELINE
# =============================================================================

@dataclass
class Pipeline:
    """A sequence of operations."""
    operations: List[str]
    
    def __post_init__(self):
        self._ops = [OPERATION_REGISTRY[name] for name in self.operations]
    
    def apply(self, n: int, max_iter: int = 100) -> Tuple[int, List[int]]:
        sequence = [n]
        seen = {n}
        current = n
        
        for _ in range(max_iter):
            for op in self._ops:
                try:
                    current = op.apply(current)
                    if current > 10**15:
                        return current, sequence
                except:
                    return current, sequence
            
            if current in seen:
                sequence.append(current)
                return current, sequence
            
            seen.add(current)
            sequence.append(current)
        
        return current, sequence
    
    def signature(self) -> str:
        return " → ".join(self.operations)
    
    def hash(self) -> str:
        return hashlib.md5(self.signature().encode()).hexdigest()[:12]
    
    def mutate(self, rng: random.Random) -> 'Pipeline':
        ops = self.operations.copy()
        mutation_type = rng.choice(['add', 'remove', 'replace', 'swap'])
        
        if mutation_type == 'add' and len(ops) < 6:
            pos = rng.randint(0, len(ops))
            new_op = rng.choice(list(OPERATION_REGISTRY.keys()))
            ops.insert(pos, new_op)
        elif mutation_type == 'remove' and len(ops) > 1:
            pos = rng.randint(0, len(ops) - 1)
            ops.pop(pos)
        elif mutation_type == 'replace' and ops:
            pos = rng.randint(0, len(ops) - 1)
            ops[pos] = rng.choice(list(OPERATION_REGISTRY.keys()))
        elif mutation_type == 'swap' and len(ops) >= 2:
            i, j = rng.sample(range(len(ops)), 2)
            ops[i], ops[j] = ops[j], ops[i]
        
        return Pipeline(ops)
    
    @staticmethod
    def crossover(p1: 'Pipeline', p2: 'Pipeline', rng: random.Random) -> 'Pipeline':
        ops1, ops2 = p1.operations, p2.operations
        if len(ops1) == 0:
            return Pipeline(ops2[:])
        if len(ops2) == 0:
            return Pipeline(ops1[:])
        
        cut1 = rng.randint(0, len(ops1))
        cut2 = rng.randint(0, len(ops2))
        
        new_ops = ops1[:cut1] + ops2[cut2:]
        if len(new_ops) == 0:
            new_ops = [rng.choice(list(OPERATION_REGISTRY.keys()))]
        if len(new_ops) > 6:
            new_ops = new_ops[:6]
        
        return Pipeline(new_ops)
    
    @staticmethod
    def random(rng: random.Random, max_length: int = 4) -> 'Pipeline':
        length = rng.randint(1, max_length)
        ops = [rng.choice(list(OPERATION_REGISTRY.keys())) for _ in range(length)]
        return Pipeline(ops)


# =============================================================================
# PATTERN DETECTION
# =============================================================================

class PatternType(Enum):
    FIXED_POINT = auto()
    CYCLE = auto()
    UNIVERSAL_CONSTANT = auto()
    PARTIAL_CONSTANT = auto()
    DIVERGENT = auto()
    CHAOTIC = auto()


@dataclass
class DiscoveredPattern:
    pipeline: Pipeline
    pattern_type: PatternType
    attractor_value: Optional[int] = None
    cycle_values: Optional[List[int]] = None
    convergence_rate: float = 0.0
    avg_steps_to_converge: float = 0.0
    tested_range: Tuple[int, int] = (0, 0)
    num_tested: int = 0
    interestingness_score: float = 0.0
    novelty_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            'pipeline': self.pipeline.signature(),
            'pattern_type': self.pattern_type.name,
            'attractor_value': self.attractor_value,
            'cycle_values': self.cycle_values,
            'convergence_rate': self.convergence_rate,
            'avg_steps': self.avg_steps_to_converge,
            'tested_range': self.tested_range,
            'num_tested': self.num_tested,
            'interestingness': self.interestingness_score,
            'novelty': self.novelty_score,
            'timestamp': self.timestamp,
        }


class PatternDetector:
    def __init__(self, config: DiscoveryConfig):
        self.config = config
    
    def analyze(self, pipeline: Pipeline, numbers: np.ndarray) -> DiscoveredPattern:
        endpoints = Counter()
        cycles = Counter()
        steps_to_converge = []
        divergent_count = 0
        
        for n in numbers:
            n = int(n)
            final, sequence = pipeline.apply(n, max_iter=self.config.max_iterations_per_pipeline)
            
            if final > 10**12:
                divergent_count += 1
                continue
            
            if len(sequence) >= 2 and sequence[-1] in sequence[:-1]:
                idx = sequence.index(sequence[-1])
                cycle = tuple(sequence[idx:-1])
                cycles[cycle] += 1
                steps_to_converge.append(idx)
            else:
                endpoints[final] += 1
                steps_to_converge.append(len(sequence) - 1)
        
        total = len(numbers)
        
        if divergent_count > total * 0.9:
            return DiscoveredPattern(
                pipeline=pipeline,
                pattern_type=PatternType.DIVERGENT,
                tested_range=(int(numbers.min()), int(numbers.max())),
                num_tested=total,
            )
        
        if endpoints:
            most_common_endpoint, count = endpoints.most_common(1)[0]
            if count >= total * 0.95:
                return DiscoveredPattern(
                    pipeline=pipeline,
                    pattern_type=PatternType.UNIVERSAL_CONSTANT,
                    attractor_value=most_common_endpoint,
                    convergence_rate=count / total,
                    avg_steps_to_converge=np.mean(steps_to_converge) if steps_to_converge else 0,
                    tested_range=(int(numbers.min()), int(numbers.max())),
                    num_tested=total,
                )
            elif count >= total * 0.5:
                return DiscoveredPattern(
                    pipeline=pipeline,
                    pattern_type=PatternType.PARTIAL_CONSTANT,
                    attractor_value=most_common_endpoint,
                    convergence_rate=count / total,
                    avg_steps_to_converge=np.mean(steps_to_converge) if steps_to_converge else 0,
                    tested_range=(int(numbers.min()), int(numbers.max())),
                    num_tested=total,
                )
        
        if cycles:
            most_common_cycle, count = cycles.most_common(1)[0]
            if count >= total * 0.5:
                return DiscoveredPattern(
                    pipeline=pipeline,
                    pattern_type=PatternType.CYCLE,
                    cycle_values=list(most_common_cycle),
                    convergence_rate=count / total,
                    avg_steps_to_converge=np.mean(steps_to_converge) if steps_to_converge else 0,
                    tested_range=(int(numbers.min()), int(numbers.max())),
                    num_tested=total,
                )
        
        return DiscoveredPattern(
            pipeline=pipeline,
            pattern_type=PatternType.CHAOTIC,
            tested_range=(int(numbers.min()), int(numbers.max())),
            num_tested=total,
        )
    
    def score_interestingness(self, pattern: DiscoveredPattern) -> float:
        score = 0.0
        
        if pattern.pattern_type == PatternType.UNIVERSAL_CONSTANT:
            score += 50.0
            score += 30.0 * pattern.convergence_rate
        elif pattern.pattern_type == PatternType.CYCLE:
            score += 30.0
            score += 20.0 * pattern.convergence_rate
        elif pattern.pattern_type == PatternType.PARTIAL_CONSTANT:
            score += 20.0 * pattern.convergence_rate
        
        if pattern.attractor_value is not None:
            if pattern.attractor_value > 0:
                score += 10.0 / (1 + math.log10(pattern.attractor_value))
            s = str(pattern.attractor_value)
            if s == s[::-1]:
                score += 15.0
            if len(set(s)) == 1:
                score += 10.0
        
        if pattern.cycle_values:
            cycle_len = len(pattern.cycle_values)
            if cycle_len <= 3:
                score += 15.0
            elif cycle_len <= 6:
                score += 10.0
        
        pipeline_len = len(pattern.pipeline.operations)
        score += 10.0 / pipeline_len
        
        if pattern.avg_steps_to_converge > 0:
            score += 5.0 / (1 + pattern.avg_steps_to_converge / 10)
        
        return score


# =============================================================================
# DISCOVERY ENGINE
# =============================================================================

class DiscoveryEngine:
    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.detector = PatternDetector(config)
        self.rng = random.Random(42)
        
        self.population: List[Pipeline] = []
        self.discoveries: List[DiscoveredPattern] = []
        self.novelty_archive: Set[str] = set()
        
        self.op_successes: Dict[str, float] = defaultdict(lambda: 1.0)
        self.op_failures: Dict[str, float] = defaultdict(lambda: 1.0)
        
        self.generation = 0
        self.total_evaluations = 0
        
        self._init_database()
    
    def _init_database(self):
        self.db_path = Path(self.config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline TEXT,
                pattern_type TEXT,
                attractor_value INTEGER,
                cycle_values TEXT,
                convergence_rate REAL,
                avg_steps REAL,
                tested_range_min INTEGER,
                tested_range_max INTEGER,
                num_tested INTEGER,
                interestingness REAL,
                novelty REAL,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def _save_discovery(self, pattern: DiscoveredPattern):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''
            INSERT INTO discoveries 
            (pipeline, pattern_type, attractor_value, cycle_values, 
             convergence_rate, avg_steps, tested_range_min, tested_range_max,
             num_tested, interestingness, novelty, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pipeline.signature(),
            pattern.pattern_type.name,
            pattern.attractor_value,
            json.dumps(pattern.cycle_values) if pattern.cycle_values else None,
            pattern.convergence_rate,
            pattern.avg_steps_to_converge,
            pattern.tested_range[0],
            pattern.tested_range[1],
            pattern.num_tested,
            pattern.interestingness_score,
            pattern.novelty_score,
            pattern.timestamp,
        ))
        conn.commit()
        conn.close()
    
    def _thompson_sample_operation(self) -> str:
        samples = {}
        for op_name in OPERATION_REGISTRY.keys():
            alpha = self.op_successes[op_name]
            beta = self.op_failures[op_name]
            samples[op_name] = self.rng.betavariate(alpha, beta)
        return max(samples, key=samples.get)
    
    def _update_thompson(self, pipeline: Pipeline, success: bool):
        for op_name in pipeline.operations:
            if success:
                self.op_successes[op_name] += 1.0
            else:
                self.op_failures[op_name] += 0.5
    
    def _compute_novelty(self, pattern: DiscoveredPattern) -> float:
        sig = pattern.pipeline.hash()
        if sig in self.novelty_archive:
            return 0.0
        
        attractor_novelty = 1.0
        for disc in self.discoveries[-100:]:
            if disc.attractor_value == pattern.attractor_value:
                attractor_novelty *= 0.5
            if disc.pipeline.signature() == pattern.pipeline.signature():
                attractor_novelty *= 0.1
        
        return attractor_novelty
    
    def generate_test_numbers(self, digit_range: Tuple[int, int], count: int) -> np.ndarray:
        min_digits, max_digits = digit_range
        if min_digits == max_digits:
            low = 10 ** (min_digits - 1)
            high = 10 ** min_digits - 1
        else:
            low = 10 ** (min_digits - 1)
            high = 10 ** max_digits - 1
        return np.random.randint(low, high + 1, size=count, dtype=np.int64)
    
    def evaluate_pipeline(self, pipeline: Pipeline) -> Tuple[DiscoveredPattern, float]:
        best_pattern = None
        best_score = 0.0
        
        for digit_range in self.config.digit_ranges:
            numbers = self.generate_test_numbers(digit_range, self.config.min_test_numbers)
            pattern = self.detector.analyze(pipeline, numbers)
            score = self.detector.score_interestingness(pattern)
            
            if score > 30.0:
                numbers = self.generate_test_numbers(digit_range, self.config.max_test_numbers)
                pattern = self.detector.analyze(pipeline, numbers)
                score = self.detector.score_interestingness(pattern)
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        if best_pattern:
            best_pattern.interestingness_score = best_score
            best_pattern.novelty_score = self._compute_novelty(best_pattern)
        
        self.total_evaluations += 1
        return best_pattern, best_score
    
    def initialize_population(self):
        self.population = []
        
        known_interesting = [
            Pipeline(['kaprekar_step']),
            Pipeline(['truc_1089']),
            Pipeline(['sub_reverse', 'add_reverse']),
            Pipeline(['happy_step']),
            Pipeline(['digitsum']),
            Pipeline(['collatz_step']),
        ]
        self.population.extend(known_interesting)
        
        while len(self.population) < self.config.population_size:
            length = self.rng.randint(1, self.config.max_pipeline_length)
            ops = [self._thompson_sample_operation() for _ in range(length)]
            self.population.append(Pipeline(ops))
    
    def evolve_generation(self) -> List[Tuple[DiscoveredPattern, float]]:
        results = []
        for i, pipeline in enumerate(self.population):
            if i % 20 == 0:
                print(f"\r  Evaluating {i}/{len(self.population)}...", end="", flush=True)
            pattern, score = self.evaluate_pipeline(pipeline)
            results.append((pipeline, pattern, score))
        print(f"\r  Evaluated {len(self.population)} pipelines.       ")
        
        results.sort(
            key=lambda x: x[2] + self.config.novelty_weight * (x[1].novelty_score if x[1] else 0),
            reverse=True
        )
        
        threshold = results[len(results) // 4][2]
        for pipeline, pattern, score in results:
            self._update_thompson(pipeline, score > threshold)
        
        discoveries_this_gen = []
        for pipeline, pattern, score in results:
            if pattern and score > 25.0:
                if pattern.pipeline.hash() not in self.novelty_archive:
                    self.novelty_archive.add(pattern.pipeline.hash())
                    self.discoveries.append(pattern)
                    self._save_discovery(pattern)
                    discoveries_this_gen.append((pattern, score))
        
        elite_count = int(self.config.population_size * self.config.elite_fraction)
        new_population = [r[0] for r in results[:elite_count]]
        
        while len(new_population) < self.config.population_size:
            if self.rng.random() < self.config.crossover_rate and len(new_population) >= 2:
                p1, p2 = self.rng.sample(new_population[:elite_count * 2], 2)
                child = Pipeline.crossover(p1, p2, self.rng)
            else:
                parent = self.rng.choice(new_population[:elite_count * 2])
                child = parent.mutate(self.rng) if self.rng.random() < self.config.mutation_rate else parent
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return discoveries_this_gen
    
    def run(self, num_generations: int = 100, verbose: bool = True):
        print("=" * 70)
        print("🔬 SYNTRIAD SYMMETRY DISCOVERY ENGINE")
        print("=" * 70)
        print(f"Population: {self.config.population_size}")
        print(f"Operations: {len(OPERATION_REGISTRY)}")
        print(f"Digit ranges: {self.config.digit_ranges}")
        print("=" * 70)
        
        self.initialize_population()
        start_time = time.time()
        
        for gen in range(num_generations):
            gen_start = time.time()
            discoveries = self.evolve_generation()
            gen_time = time.time() - gen_start
            
            if verbose and (gen % self.config.log_interval == 0 or discoveries):
                print(f"\n[Gen {gen:4d}] Time: {gen_time:.1f}s | Evaluations: {self.total_evaluations:,}")
                print(f"         Total discoveries: {len(self.discoveries)}")
                
                for pattern, score in discoveries:
                    print(f"\n  🎯 NEW DISCOVERY (score: {score:.1f})")
                    print(f"     Pipeline: {pattern.pipeline.signature()}")
                    print(f"     Type: {pattern.pattern_type.name}")
                    if pattern.attractor_value is not None:
                        print(f"     Attractor: {pattern.attractor_value}")
                    if pattern.cycle_values:
                        print(f"     Cycle: {pattern.cycle_values}")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"Total time: {total_time:.1f}s")
        print(f"Discoveries: {len(self.discoveries)}")
        
        top_discoveries = sorted(self.discoveries, key=lambda x: x.interestingness_score, reverse=True)[:10]
        
        print("\n🏆 TOP 10 DISCOVERIES:")
        for i, d in enumerate(top_discoveries, 1):
            print(f"{i:2d}. {d.pipeline.signature()}")
            print(f"    Type: {d.pattern_type.name} | Score: {d.interestingness_score:.1f}")
            if d.attractor_value is not None:
                print(f"    Attractor: {d.attractor_value}")
        
        return self.discoveries


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTRIAD Symmetry Discovery Engine')
    parser.add_argument('--generations', type=int, default=20)
    parser.add_argument('--population', type=int, default=30)
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--db', type=str, default='discoveries.db')
    args = parser.parse_args()
    
    config = DiscoveryConfig(
        num_cpu_workers=args.workers,
        population_size=args.population,
        db_path=args.db,
    )
    
    engine = DiscoveryEngine(config)
    discoveries = engine.run(num_generations=args.generations)
    
    print(f"\n✅ Done! {len(discoveries)} patterns discovered.")


if __name__ == "__main__":
    main()
