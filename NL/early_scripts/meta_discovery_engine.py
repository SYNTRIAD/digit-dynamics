#!/usr/bin/env python3
"""
SYNTRIAD Meta-Discovery Engine v2.0
====================================

Een zelf-verbeterend, iteratief onderzoekssysteem dat:
1. Triviale patronen filtert en penaliseert
2. Nieuwe operaties dynamisch genereert
3. Zichzelf herprogrammeert op basis van resultaten
4. Multi-dimensionale zoekstrategieën combineert

Auteur: SYNTRIAD Research
"""

import numpy as np
import json
import time
import sqlite3
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable
from collections import Counter, defaultdict
from pathlib import Path
from abc import ABC, abstractmethod
import itertools

# Import base operations
from symmetry_discovery_engine import (
    Operation, OPERATION_REGISTRY, PatternType, 
    DiscoveredPattern, DiscoveryConfig
)
import hashlib


# =============================================================================
# NIEUWE OPERATIES - Dynamisch gegenereerd
# =============================================================================

class ModuloOp(Operation):
    """Modulo met configureerbare basis."""
    def __init__(self, base: int = 9):
        self.base = base
    
    @property
    def name(self) -> str:
        return f"mod_{self.base}"
    
    def apply(self, n: int) -> int:
        return abs(n) % self.base if self.base > 0 else 0


class DigitPowerSumOp(Operation):
    """Som van cijfers tot macht p."""
    def __init__(self, power: int = 3):
        self.power = power
    
    @property
    def name(self) -> str:
        return f"digit_pow_{self.power}"
    
    def apply(self, n: int) -> int:
        return sum(int(d)**self.power for d in str(abs(n)))


class DigitProductChainOp(Operation):
    """Herhaald digit product tot 1 cijfer (multiplicative persistence)."""
    @property
    def name(self) -> str:
        return "persistence"
    
    def apply(self, n: int) -> int:
        steps = 0
        while n >= 10 and steps < 20:
            prod = 1
            for d in str(n):
                prod *= int(d)
            n = prod
            steps += 1
        return steps  # Return persistence, not final value


class DigitSortDiffOp(Operation):
    """Verschil tussen gesorteerd desc en asc (generalized Kaprekar)."""
    @property
    def name(self) -> str:
        return "sort_diff"
    
    def apply(self, n: int) -> int:
        s = str(abs(n))
        desc = int(''.join(sorted(s, reverse=True)))
        asc_str = ''.join(sorted(s)).lstrip('0') or '0'
        asc = int(asc_str)
        return desc - asc


class DigitCycleOp(Operation):
    """Cyclische permutatie van cijfers."""
    def __init__(self, shift: int = 1):
        self.shift = shift
    
    @property
    def name(self) -> str:
        return f"cycle_{self.shift}"
    
    def apply(self, n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        shift = self.shift % len(s)
        rotated = s[shift:] + s[:shift]
        return int(rotated.lstrip('0') or '0')


class DigitSwapOp(Operation):
    """Swap eerste en laatste cijfer."""
    @property
    def name(self) -> str:
        return "swap_ends"
    
    def apply(self, n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        swapped = s[-1] + s[1:-1] + s[0]
        return int(swapped.lstrip('0') or '0')


class DigitComplementOp(Operation):
    """9-complement van elk cijfer."""
    @property
    def name(self) -> str:
        return "complement_9"
    
    def apply(self, n: int) -> int:
        return int(''.join(str(9 - int(d)) for d in str(abs(n))).lstrip('0') or '0')


class AlternatingProductOp(Operation):
    """Alternerend product: d1 * d3 * d5... - d2 * d4 * d6..."""
    @property
    def name(self) -> str:
        return "alt_prod"
    
    def apply(self, n: int) -> int:
        digits = [int(d) for d in str(abs(n))]
        odd_prod = 1
        even_prod = 1
        for i, d in enumerate(digits):
            if d == 0:
                d = 1  # Avoid zero products
            if i % 2 == 0:
                odd_prod *= d
            else:
                even_prod *= d
        return abs(odd_prod - even_prod)


class DigitSquareDiffOp(Operation):
    """Som van kwadraatverschillen tussen opeenvolgende cijfers."""
    @property
    def name(self) -> str:
        return "sq_diff"
    
    def apply(self, n: int) -> int:
        digits = [int(d) for d in str(abs(n))]
        if len(digits) < 2:
            return 0
        return sum((digits[i] - digits[i+1])**2 for i in range(len(digits)-1))


# =============================================================================
# EXTENDED OPERATION REGISTRY
# =============================================================================

EXTENDED_OPERATIONS: Dict[str, Operation] = {
    **OPERATION_REGISTRY,
    'mod_9': ModuloOp(9),
    'mod_11': ModuloOp(11),
    'mod_7': ModuloOp(7),
    'digit_pow_3': DigitPowerSumOp(3),  # Narcissistic numbers
    'digit_pow_4': DigitPowerSumOp(4),
    'persistence': DigitProductChainOp(),
    'sort_diff': DigitSortDiffOp(),
    'cycle_2': DigitCycleOp(2),
    'swap_ends': DigitSwapOp(),
    'complement_9': DigitComplementOp(),
    'alt_prod': AlternatingProductOp(),
    'sq_diff': DigitSquareDiffOp(),
}


# =============================================================================
# EXTENDED PIPELINE - Uses EXTENDED_OPERATIONS
# =============================================================================

@dataclass
class Pipeline:
    """Extended pipeline that uses EXTENDED_OPERATIONS."""
    operations: List[str]
    
    def __post_init__(self):
        self._ops = [EXTENDED_OPERATIONS[name] for name in self.operations]
    
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


# =============================================================================
# IMPROVED SCORING - Penalizes trivial patterns
# =============================================================================

class ImprovedScorer:
    """Verbeterde scoring die triviale patronen penaliseert."""
    
    TRIVIAL_VALUES = {0, 1, 2}
    KNOWN_CONSTANTS = {495, 6174, 1089, 10890, 109890, 1098900}
    
    def __init__(self):
        self.seen_attractors: Counter = Counter()
        self.seen_pipelines: Set[str] = set()
    
    def is_trivial(self, pattern: DiscoveredPattern) -> bool:
        """Check of patroon triviaal is."""
        if pattern.attractor_value in self.TRIVIAL_VALUES:
            return True
        if pattern.cycle_values:
            if all(v in self.TRIVIAL_VALUES for v in pattern.cycle_values):
                return True
            if len(pattern.cycle_values) == 1 and pattern.cycle_values[0] in self.TRIVIAL_VALUES:
                return True
        return False
    
    def is_novel(self, pattern: DiscoveredPattern) -> bool:
        """Check of patroon nieuw is."""
        sig = pattern.pipeline.signature()
        if sig in self.seen_pipelines:
            return False
        
        # Check attractor novelty
        if pattern.attractor_value is not None:
            if self.seen_attractors[pattern.attractor_value] > 5:
                return False
        
        return True
    
    def score(self, pattern: DiscoveredPattern) -> float:
        """Bereken verbeterde score."""
        if self.is_trivial(pattern):
            return 0.0  # Triviale patronen krijgen 0
        
        score = 0.0
        
        # Base score voor patroontype
        if pattern.pattern_type == PatternType.UNIVERSAL_CONSTANT:
            score += 60.0
        elif pattern.pattern_type == PatternType.CYCLE:
            score += 40.0
        elif pattern.pattern_type == PatternType.PARTIAL_CONSTANT:
            score += 25.0
        
        # Convergentie bonus
        score += 20.0 * pattern.convergence_rate
        
        # Attractor waarde analyse
        if pattern.attractor_value is not None and pattern.attractor_value > 2:
            # Bonus voor "interessante" getallen
            s = str(pattern.attractor_value)
            
            # Palindroom bonus
            if s == s[::-1]:
                score += 20.0
            
            # Repdigit bonus (111, 222, etc)
            if len(set(s)) == 1:
                score += 15.0
            
            # Bekende constante bonus
            if pattern.attractor_value in self.KNOWN_CONSTANTS:
                score += 25.0
            
            # Novelty bonus - nieuwe attractor
            if self.seen_attractors[pattern.attractor_value] == 0:
                score += 30.0
            
            # Grootte penalty (te grote getallen zijn minder interessant)
            if pattern.attractor_value > 1_000_000:
                score -= 10.0
        
        # Cyclus analyse
        if pattern.cycle_values and len(pattern.cycle_values) > 0:
            cycle_len = len(pattern.cycle_values)
            
            # Korte cycli zijn interessanter
            if 2 <= cycle_len <= 4:
                score += 15.0
            elif cycle_len <= 8:
                score += 10.0
            
            # Cyclus met interessante getallen
            interesting_in_cycle = sum(1 for v in pattern.cycle_values 
                                       if v > 10 and str(v) == str(v)[::-1])
            score += 5.0 * interesting_in_cycle
        
        # Pipeline complexiteit
        pipeline_len = len(pattern.pipeline.operations)
        if pipeline_len == 1:
            score += 15.0  # Simpele pipelines zijn elegant
        elif pipeline_len == 2:
            score += 10.0
        elif pipeline_len > 4:
            score -= 5.0  # Te complexe pipelines penaliseren
        
        # Novelty bonus
        if self.is_novel(pattern):
            score += 20.0
        
        # Update tracking
        self.seen_pipelines.add(pattern.pipeline.signature())
        if pattern.attractor_value is not None:
            self.seen_attractors[pattern.attractor_value] += 1
        
        return max(0.0, score)


# =============================================================================
# META-DISCOVERY ENGINE
# =============================================================================

@dataclass
class MetaConfig:
    """Configuratie voor meta-discovery."""
    population_size: int = 100
    generations: int = 50
    elite_fraction: float = 0.15
    mutation_rate: float = 0.4
    crossover_rate: float = 0.3
    exploration_rate: float = 0.2  # Kans op volledig nieuwe pipeline
    max_pipeline_length: int = 4
    min_test_numbers: int = 1000
    max_test_numbers: int = 10000
    digit_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (3, 3), (4, 4), (5, 5), (6, 6)
    ])
    db_path: str = "meta_discoveries.db"
    
    # Self-improvement parameters
    adapt_mutation_rate: bool = True
    adapt_exploration_rate: bool = True
    stagnation_threshold: int = 5  # Generaties zonder verbetering


class MetaDiscoveryEngine:
    """
    Zelf-verbeterend discovery systeem.
    
    Features:
    - Dynamische operatie-selectie via Thompson sampling
    - Adaptieve mutatie/exploratie rates
    - Multi-strategie zoeken
    - Automatische parameter tuning
    """
    
    def __init__(self, config: MetaConfig = None):
        self.config = config or MetaConfig()
        self.scorer = ImprovedScorer()
        self.rng = random.Random(42)
        
        # Population
        self.population: List[Pipeline] = []
        self.discoveries: List[DiscoveredPattern] = []
        
        # Thompson sampling voor operaties
        self.op_alpha: Dict[str, float] = defaultdict(lambda: 1.0)
        self.op_beta: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Strategy tracking
        self.strategy_scores: Dict[str, float] = {
            'mutation': 1.0,
            'crossover': 1.0,
            'exploration': 1.0,
            'guided': 1.0,
        }
        
        # Self-improvement tracking
        self.generation = 0
        self.best_score_history: List[float] = []
        self.stagnation_counter = 0
        
        # Adaptive parameters
        self.current_mutation_rate = self.config.mutation_rate
        self.current_exploration_rate = self.config.exploration_rate
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        self.db_path = Path(self.config.db_path)
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS meta_discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline TEXT,
                pattern_type TEXT,
                attractor_value INTEGER,
                cycle_values TEXT,
                convergence_rate REAL,
                score REAL,
                generation INTEGER,
                strategy TEXT,
                timestamp REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS meta_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER,
                mutation_rate REAL,
                exploration_rate REAL,
                best_score REAL,
                discoveries_count INTEGER,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def _save_discovery(self, pattern: DiscoveredPattern, strategy: str):
        """Save discovery to database."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''
            INSERT INTO meta_discoveries 
            (pipeline, pattern_type, attractor_value, cycle_values, 
             convergence_rate, score, generation, strategy, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pipeline.signature(),
            pattern.pattern_type.name,
            pattern.attractor_value,
            json.dumps(pattern.cycle_values) if pattern.cycle_values else None,
            pattern.convergence_rate,
            pattern.interestingness_score,
            self.generation,
            strategy,
            time.time(),
        ))
        conn.commit()
        conn.close()
    
    def _save_parameters(self, best_score: float):
        """Save adaptive parameters."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute('''
            INSERT INTO meta_parameters 
            (generation, mutation_rate, exploration_rate, best_score, 
             discoveries_count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.generation,
            self.current_mutation_rate,
            self.current_exploration_rate,
            best_score,
            len(self.discoveries),
            time.time(),
        ))
        conn.commit()
        conn.close()
    
    def thompson_sample_operation(self) -> str:
        """Sample operatie via Thompson sampling."""
        samples = {}
        for op_name in EXTENDED_OPERATIONS.keys():
            alpha = self.op_alpha[op_name]
            beta = self.op_beta[op_name]
            samples[op_name] = self.rng.betavariate(alpha, beta)
        return max(samples, key=samples.get)
    
    def update_thompson(self, pipeline: Pipeline, success: bool):
        """Update Thompson sampling parameters."""
        for op_name in pipeline.operations:
            if success:
                self.op_alpha[op_name] += 1.0
            else:
                self.op_beta[op_name] += 0.3
    
    def generate_test_numbers(self, digit_range: Tuple[int, int], count: int) -> np.ndarray:
        """Generate test numbers."""
        min_d, max_d = digit_range
        if min_d == max_d:
            low = 10 ** (min_d - 1)
            high = 10 ** min_d - 1
        else:
            low = 10 ** (min_d - 1)
            high = 10 ** max_d - 1
        return np.random.randint(low, high + 1, size=count, dtype=np.int64)
    
    def analyze_pipeline(self, pipeline: Pipeline, numbers: np.ndarray) -> DiscoveredPattern:
        """Analyze a pipeline on test numbers."""
        endpoints = Counter()
        cycles = Counter()
        steps_list = []
        divergent = 0
        
        for n in numbers:
            n = int(n)
            try:
                final, sequence = self._apply_pipeline(pipeline, n)
            except:
                divergent += 1
                continue
            
            if final > 10**12:
                divergent += 1
                continue
            
            if len(sequence) >= 2 and sequence[-1] in sequence[:-1]:
                idx = sequence.index(sequence[-1])
                cycle = tuple(sequence[idx:-1])
                cycles[cycle] += 1
                steps_list.append(idx)
            else:
                endpoints[final] += 1
                steps_list.append(len(sequence) - 1)
        
        total = len(numbers)
        
        if divergent > total * 0.9:
            return DiscoveredPattern(
                pipeline=pipeline,
                pattern_type=PatternType.DIVERGENT,
                tested_range=(int(numbers.min()), int(numbers.max())),
                num_tested=total,
            )
        
        if endpoints:
            most_common, count = endpoints.most_common(1)[0]
            rate = count / total
            
            if rate >= 0.95:
                return DiscoveredPattern(
                    pipeline=pipeline,
                    pattern_type=PatternType.UNIVERSAL_CONSTANT,
                    attractor_value=most_common,
                    convergence_rate=rate,
                    avg_steps_to_converge=np.mean(steps_list) if steps_list else 0,
                    tested_range=(int(numbers.min()), int(numbers.max())),
                    num_tested=total,
                )
            elif rate >= 0.5:
                return DiscoveredPattern(
                    pipeline=pipeline,
                    pattern_type=PatternType.PARTIAL_CONSTANT,
                    attractor_value=most_common,
                    convergence_rate=rate,
                    avg_steps_to_converge=np.mean(steps_list) if steps_list else 0,
                    tested_range=(int(numbers.min()), int(numbers.max())),
                    num_tested=total,
                )
        
        if cycles:
            most_common_cycle, count = cycles.most_common(1)[0]
            rate = count / total
            if rate >= 0.5:
                return DiscoveredPattern(
                    pipeline=pipeline,
                    pattern_type=PatternType.CYCLE,
                    cycle_values=list(most_common_cycle),
                    convergence_rate=rate,
                    avg_steps_to_converge=np.mean(steps_list) if steps_list else 0,
                    tested_range=(int(numbers.min()), int(numbers.max())),
                    num_tested=total,
                )
        
        return DiscoveredPattern(
            pipeline=pipeline,
            pattern_type=PatternType.CHAOTIC,
            tested_range=(int(numbers.min()), int(numbers.max())),
            num_tested=total,
        )
    
    def _apply_pipeline(self, pipeline: Pipeline, n: int, max_iter: int = 100) -> Tuple[int, List[int]]:
        """Apply pipeline to number."""
        sequence = [n]
        seen = {n}
        current = n
        
        ops = [EXTENDED_OPERATIONS[name] for name in pipeline.operations]
        
        for _ in range(max_iter):
            for op in ops:
                current = op.apply(current)
                if current > 10**15:
                    return current, sequence
            
            if current in seen:
                sequence.append(current)
                return current, sequence
            
            seen.add(current)
            sequence.append(current)
        
        return current, sequence
    
    def create_random_pipeline(self, max_length: int = None) -> Pipeline:
        """Create random pipeline using Thompson sampling."""
        max_length = max_length or self.config.max_pipeline_length
        length = self.rng.randint(1, max_length)
        ops = [self.thompson_sample_operation() for _ in range(length)]
        return Pipeline(ops)
    
    def create_guided_pipeline(self) -> Pipeline:
        """Create pipeline guided by successful patterns."""
        if not self.discoveries:
            return self.create_random_pipeline()
        
        # Select from top discoveries
        top = sorted(self.discoveries, key=lambda x: x.interestingness_score, reverse=True)[:10]
        base = self.rng.choice(top).pipeline
        
        # Mutate slightly
        return self._mutate_pipeline(base)
    
    def _mutate_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Mutate a pipeline."""
        ops = list(pipeline.operations)
        mutation_type = self.rng.choice(['add', 'remove', 'replace', 'swap', 'insert'])
        
        if mutation_type == 'add' and len(ops) < self.config.max_pipeline_length:
            ops.append(self.thompson_sample_operation())
        elif mutation_type == 'remove' and len(ops) > 1:
            ops.pop(self.rng.randint(0, len(ops) - 1))
        elif mutation_type == 'replace' and ops:
            idx = self.rng.randint(0, len(ops) - 1)
            ops[idx] = self.thompson_sample_operation()
        elif mutation_type == 'swap' and len(ops) >= 2:
            i, j = self.rng.sample(range(len(ops)), 2)
            ops[i], ops[j] = ops[j], ops[i]
        elif mutation_type == 'insert' and len(ops) < self.config.max_pipeline_length:
            pos = self.rng.randint(0, len(ops))
            ops.insert(pos, self.thompson_sample_operation())
        
        return Pipeline(ops)
    
    def _crossover(self, p1: Pipeline, p2: Pipeline) -> Pipeline:
        """Crossover two pipelines."""
        ops1, ops2 = list(p1.operations), list(p2.operations)
        
        if not ops1:
            return Pipeline(ops2[:])
        if not ops2:
            return Pipeline(ops1[:])
        
        cut1 = self.rng.randint(0, len(ops1))
        cut2 = self.rng.randint(0, len(ops2))
        
        new_ops = ops1[:cut1] + ops2[cut2:]
        
        if not new_ops:
            new_ops = [self.thompson_sample_operation()]
        if len(new_ops) > self.config.max_pipeline_length:
            new_ops = new_ops[:self.config.max_pipeline_length]
        
        return Pipeline(new_ops)
    
    def select_strategy(self) -> str:
        """Select breeding strategy via Thompson sampling."""
        total = sum(self.strategy_scores.values())
        r = self.rng.random() * total
        cumsum = 0
        for strategy, score in self.strategy_scores.items():
            cumsum += score
            if r <= cumsum:
                return strategy
        return 'mutation'
    
    def update_strategy_scores(self, strategy: str, success: bool):
        """Update strategy success tracking."""
        if success:
            self.strategy_scores[strategy] *= 1.1
        else:
            self.strategy_scores[strategy] *= 0.95
        
        # Normalize
        total = sum(self.strategy_scores.values())
        for k in self.strategy_scores:
            self.strategy_scores[k] /= total
    
    def adapt_parameters(self):
        """Adapt mutation and exploration rates based on progress."""
        if not self.config.adapt_mutation_rate:
            return
        
        if len(self.best_score_history) < 2:
            return
        
        # Check for stagnation
        recent_best = max(self.best_score_history[-5:]) if len(self.best_score_history) >= 5 else self.best_score_history[-1]
        older_best = max(self.best_score_history[:-5]) if len(self.best_score_history) > 5 else 0
        
        if recent_best <= older_best * 1.01:  # Less than 1% improvement
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # Adapt based on stagnation
        if self.stagnation_counter >= self.config.stagnation_threshold:
            # Increase exploration
            self.current_exploration_rate = min(0.5, self.current_exploration_rate * 1.2)
            self.current_mutation_rate = min(0.6, self.current_mutation_rate * 1.1)
            print(f"  📈 Adapting: exploration={self.current_exploration_rate:.2f}, mutation={self.current_mutation_rate:.2f}")
            self.stagnation_counter = 0
        elif self.stagnation_counter == 0 and self.generation > 10:
            # Decrease exploration (exploitation phase)
            self.current_exploration_rate = max(0.1, self.current_exploration_rate * 0.95)
    
    def initialize_population(self):
        """Initialize population with diverse pipelines."""
        self.population = []
        
        # Seed with known interesting pipelines
        seeds = [
            ['kaprekar_step'],
            ['truc_1089'],
            ['happy_step'],
            ['digit_pow_3'],  # Narcissistic numbers
            ['persistence'],  # Multiplicative persistence
            ['sort_diff'],
            ['complement_9', 'add_reverse'],
            ['digit_pow_4'],
            ['sq_diff', 'happy_step'],
        ]
        
        for ops in seeds:
            try:
                self.population.append(Pipeline(ops))
            except:
                pass
        
        # Fill with random pipelines
        while len(self.population) < self.config.population_size:
            self.population.append(self.create_random_pipeline())
    
    def evaluate_population(self) -> List[Tuple[Pipeline, DiscoveredPattern, float]]:
        """Evaluate all pipelines in population."""
        results = []
        
        for i, pipeline in enumerate(self.population):
            if i % 20 == 0:
                print(f"\r  Evaluating {i}/{len(self.population)}...", end="", flush=True)
            
            best_pattern = None
            best_score = 0.0
            
            for digit_range in self.config.digit_ranges:
                numbers = self.generate_test_numbers(digit_range, self.config.min_test_numbers)
                pattern = self.analyze_pipeline(pipeline, numbers)
                score = self.scorer.score(pattern)
                
                # If promising, test with more numbers
                if score > 30.0:
                    numbers = self.generate_test_numbers(digit_range, self.config.max_test_numbers)
                    pattern = self.analyze_pipeline(pipeline, numbers)
                    score = self.scorer.score(pattern)
                
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
            
            if best_pattern:
                best_pattern.interestingness_score = best_score
            
            results.append((pipeline, best_pattern, best_score))
        
        print(f"\r  Evaluated {len(self.population)} pipelines.       ")
        return results
    
    def evolve_generation(self) -> List[Tuple[DiscoveredPattern, float, str]]:
        """Evolve one generation."""
        results = self.evaluate_population()
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Update Thompson sampling
        threshold = results[len(results) // 3][2]
        for pipeline, pattern, score in results:
            self.update_thompson(pipeline, score > threshold)
        
        # Record discoveries - lower threshold, track all interesting patterns
        discoveries_this_gen = []
        for pipeline, pattern, score in results:
            if pattern and score > 20.0:  # Lower threshold
                sig = pattern.pipeline.signature()
                is_trivial = self.scorer.is_trivial(pattern)
                
                # Save non-trivial discoveries
                if not is_trivial and sig not in [d.pipeline.signature() for d in self.discoveries]:
                    self.discoveries.append(pattern)
                    strategy = 'initial' if self.generation == 0 else 'evolved'
                    self._save_discovery(pattern, strategy)
                    discoveries_this_gen.append((pattern, score, strategy))
        
        # Track best score
        best_score = results[0][2] if results else 0
        self.best_score_history.append(best_score)
        
        # Adapt parameters
        self.adapt_parameters()
        self._save_parameters(best_score)
        
        # Create new population
        elite_count = int(self.config.population_size * self.config.elite_fraction)
        new_population = [r[0] for r in results[:elite_count]]
        
        while len(new_population) < self.config.population_size:
            strategy = self.select_strategy()
            
            if strategy == 'exploration' or self.rng.random() < self.current_exploration_rate:
                child = self.create_random_pipeline()
                strategy = 'exploration'
            elif strategy == 'guided' and self.discoveries:
                child = self.create_guided_pipeline()
            elif strategy == 'crossover' and len(new_population) >= 2:
                p1, p2 = self.rng.sample(new_population[:elite_count * 2], 2)
                child = self._crossover(p1, p2)
            else:
                parent = self.rng.choice(new_population[:elite_count * 2])
                child = self._mutate_pipeline(parent) if self.rng.random() < self.current_mutation_rate else parent
                strategy = 'mutation'
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return discoveries_this_gen
    
    def run(self, verbose: bool = True):
        """Run the meta-discovery engine."""
        print("█" * 70)
        print("  SYNTRIAD META-DISCOVERY ENGINE v2.0")
        print("█" * 70)
        print(f"Population: {self.config.population_size}")
        print(f"Generations: {self.config.generations}")
        print(f"Operations: {len(EXTENDED_OPERATIONS)}")
        print(f"Digit ranges: {self.config.digit_ranges}")
        print("█" * 70)
        
        self.initialize_population()
        start_time = time.time()
        
        for gen in range(self.config.generations):
            gen_start = time.time()
            discoveries = self.evolve_generation()
            gen_time = time.time() - gen_start
            
            if verbose:
                print(f"\n[Gen {gen:4d}] Time: {gen_time:.1f}s | Best: {self.best_score_history[-1]:.1f}")
                print(f"         Discoveries: {len(self.discoveries)} | Stagnation: {self.stagnation_counter}")
                
                for pattern, score, strategy in discoveries[:3]:
                    print(f"\n  🎯 NEW ({strategy}): score={score:.1f}")
                    print(f"     Pipeline: {pattern.pipeline.signature()}")
                    print(f"     Type: {pattern.pattern_type.name}")
                    if pattern.attractor_value is not None:
                        print(f"     Attractor: {pattern.attractor_value}")
                    if pattern.cycle_values:
                        print(f"     Cycle: {pattern.cycle_values[:5]}{'...' if len(pattern.cycle_values) > 5 else ''}")
        
        total_time = time.time() - start_time
        
        # Summary
        print("\n" + "█" * 70)
        print("  SUMMARY")
        print("█" * 70)
        print(f"Total time: {total_time:.1f}s")
        print(f"Total discoveries: {len(self.discoveries)}")
        
        # Top discoveries (non-trivial)
        top = sorted(
            [d for d in self.discoveries if not self.scorer.is_trivial(d)],
            key=lambda x: x.interestingness_score,
            reverse=True
        )[:15]
        
        print("\n🏆 TOP 15 NON-TRIVIAL DISCOVERIES:")
        for i, d in enumerate(top, 1):
            print(f"{i:2d}. {d.pipeline.signature()}")
            print(f"    Type: {d.pattern_type.name} | Score: {d.interestingness_score:.1f}")
            if d.attractor_value is not None:
                print(f"    Attractor: {d.attractor_value}")
            if d.cycle_values:
                print(f"    Cycle: {d.cycle_values[:5]}")
        
        # Operation effectiveness
        print("\n📊 TOP OPERATIONS (Thompson sampling):")
        op_scores = {op: self.op_alpha[op] / (self.op_alpha[op] + self.op_beta[op]) 
                     for op in EXTENDED_OPERATIONS.keys()}
        for op, score in sorted(op_scores.items(), key=lambda x: -x[1])[:10]:
            print(f"   {op:25s}: {score:.3f}")
        
        return self.discoveries


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTRIAD Meta-Discovery Engine')
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--population', type=int, default=80)
    parser.add_argument('--db', type=str, default='meta_discoveries.db')
    args = parser.parse_args()
    
    config = MetaConfig(
        population_size=args.population,
        generations=args.generations,
        db_path=args.db,
    )
    
    engine = MetaDiscoveryEngine(config)
    discoveries = engine.run()
    
    print(f"\n✅ Done! {len(discoveries)} patterns discovered.")


if __name__ == "__main__":
    main()
