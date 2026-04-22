#!/usr/bin/env python3
"""
SYNTRIAD Autonomous Researcher v3.0
====================================

A fully autonomous, self-reprogramming research system that:
1. Continuously generates new operations and combinations
2. Self-optimizes based on discoveries
3. Performs in-depth analyses on interesting patterns
4. Formulates and tests new hypotheses

This is the "AI-driven building block" for autonomous mathematical research.
"""

import numpy as np
import json
import time
import sqlite3
import random
import math
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable, Any
from collections import Counter, defaultdict
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
import itertools

from meta_discovery_engine import (
    Pipeline, EXTENDED_OPERATIONS, Operation,
    ImprovedScorer, PatternType, DiscoveredPattern
)


# =============================================================================
# DYNAMIC OPERATION GENERATOR
# =============================================================================

class DynamicOperationFactory:
    """Generates new operations dynamically based on templates."""
    
    TEMPLATES = {
        'power_sum': lambda p: type('PowerSum', (Operation,), {
            'power': p,
            'name': property(lambda self: f'pow_sum_{p}'),
            'apply': lambda self, n: sum(int(d)**p for d in str(abs(n)))
        })(),
        'modulo': lambda m: type('Modulo', (Operation,), {
            'mod': m,
            'name': property(lambda self: f'mod_{m}'),
            'apply': lambda self, n: abs(n) % m if m > 0 else 0
        })(),
        'digit_filter': lambda pred_str: type('DigitFilter', (Operation,), {
            'pred': pred_str,
            'name': property(lambda self: f'filter_{pred_str}'),
            'apply': lambda self, n: int(''.join(d for d in str(abs(n)) if eval(pred_str.replace('d', d))) or '0')
        })(),
        'base_convert': lambda b: type('BaseConvert', (Operation,), {
            'base': b,
            'name': property(lambda self: f'base_{b}'),
            'apply': lambda self, n: self._convert(abs(n), b),
            '_convert': lambda self, n, b: sum(int(d) for d in np.base_repr(n, b)) if n > 0 else 0
        })(),
    }
    
    def __init__(self):
        self.generated_ops: Dict[str, Operation] = {}
    
    def generate_power_sum(self, power: int) -> Operation:
        """Generate sum-of-powers operation."""
        name = f'pow_sum_{power}'
        if name not in self.generated_ops:
            class PowerSumOp(Operation):
                def __init__(self, p):
                    self.p = p
                @property
                def name(self):
                    return f'pow_sum_{self.p}'
                def apply(self, n):
                    return sum(int(d)**self.p for d in str(abs(n)))
            self.generated_ops[name] = PowerSumOp(power)
        return self.generated_ops[name]
    
    def generate_modulo(self, mod: int) -> Operation:
        """Generate modulo operation."""
        name = f'mod_{mod}'
        if name not in self.generated_ops:
            class ModOp(Operation):
                def __init__(self, m):
                    self.m = m
                @property
                def name(self):
                    return f'mod_{self.m}'
                def apply(self, n):
                    return abs(n) % self.m if self.m > 0 else 0
            self.generated_ops[name] = ModOp(mod)
        return self.generated_ops[name]
    
    def generate_weighted_sum(self, weights: List[int]) -> Operation:
        """Generate weighted digit sum."""
        name = f'weighted_{"".join(map(str, weights))}'
        if name not in self.generated_ops:
            class WeightedSumOp(Operation):
                def __init__(self, w):
                    self.w = w
                @property
                def name(self):
                    return f'weighted_{len(self.w)}'
                def apply(self, n):
                    digits = [int(d) for d in str(abs(n))]
                    return sum(d * self.w[i % len(self.w)] for i, d in enumerate(digits))
            self.generated_ops[name] = WeightedSumOp(weights)
        return self.generated_ops[name]
    
    def get_all_ops(self) -> Dict[str, Operation]:
        """Return all operations (base + generated)."""
        return {**EXTENDED_OPERATIONS, **self.generated_ops}


# =============================================================================
# HYPOTHESIS ENGINE
# =============================================================================

@dataclass
class Hypothesis:
    """A mathematical hypothesis to test."""
    description: str
    pipeline: Pipeline
    expected_pattern: str  # 'constant', 'cycle', 'divergent'
    expected_value: Optional[int] = None
    digit_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [(3,3), (4,4), (5,5)])
    confidence: float = 0.0
    tested: bool = False
    confirmed: bool = False
    
    def to_dict(self) -> dict:
        return {
            'description': self.description,
            'pipeline': self.pipeline.signature(),
            'expected_pattern': self.expected_pattern,
            'expected_value': self.expected_value,
            'confidence': self.confidence,
            'confirmed': self.confirmed,
        }


class HypothesisEngine:
    """Generates and tests mathematical hypotheses."""
    
    def __init__(self, op_factory: DynamicOperationFactory):
        self.op_factory = op_factory
        self.hypotheses: List[Hypothesis] = []
        self.confirmed: List[Hypothesis] = []
        self.refuted: List[Hypothesis] = []
    
    def generate_hypotheses(self, discoveries: List[DiscoveredPattern]) -> List[Hypothesis]:
        """Generate new hypotheses based on discoveries."""
        new_hypotheses = []
        
        # Hypothesis 1: If pipeline P converges to constant C,
        # then P' (variation) may also converge to C or related C'
        for disc in discoveries[:20]:
            if disc.attractor_value and disc.attractor_value > 10:
                # Variation: add operation at the beginning
                for op_name in ['reverse', 'complement_9', 'swap_ends']:
                    new_ops = [op_name] + list(disc.pipeline.operations)
                    try:
                        new_pipeline = Pipeline(new_ops)
                        hyp = Hypothesis(
                            description=f"Variation of {disc.pipeline.signature()} with {op_name} prefix",
                            pipeline=new_pipeline,
                            expected_pattern='constant' if disc.pattern_type == PatternType.UNIVERSAL_CONSTANT else 'cycle',
                            expected_value=disc.attractor_value,
                        )
                        new_hypotheses.append(hyp)
                    except:
                        pass
        
        # Hypothesis 2: Combinations of successful operations
        successful_ops = Counter()
        for disc in discoveries:
            for op in disc.pipeline.operations:
                if disc.interestingness_score > 50:
                    successful_ops[op] += 1
        
        top_ops = [op for op, _ in successful_ops.most_common(5)]
        for combo in itertools.combinations(top_ops, 2):
            try:
                new_pipeline = Pipeline(list(combo))
                hyp = Hypothesis(
                    description=f"Combination of top operations: {combo}",
                    pipeline=new_pipeline,
                    expected_pattern='cycle',
                )
                new_hypotheses.append(hyp)
            except:
                pass
        
        # Hypothesis 3: Power-sum variations
        for power in [5, 6, 7]:
            op = self.op_factory.generate_power_sum(power)
            try:
                new_pipeline = Pipeline([op.name])
                hyp = Hypothesis(
                    description=f"Sum of {power}th powers of digits",
                    pipeline=new_pipeline,
                    expected_pattern='cycle',
                )
                new_hypotheses.append(hyp)
            except:
                pass
        
        self.hypotheses.extend(new_hypotheses)
        return new_hypotheses
    
    def test_hypothesis(self, hyp: Hypothesis, sample_size: int = 5000) -> bool:
        """Test a hypothesis."""
        all_ops = self.op_factory.get_all_ops()
        
        results = []
        for digit_range in hyp.digit_ranges:
            min_n, max_n = 10**(digit_range[0]-1), 10**digit_range[1] - 1
            numbers = np.random.randint(min_n, max_n + 1, size=sample_size)
            
            endpoints = Counter()
            for n in numbers:
                current = int(n)
                seen = set()
                for _ in range(100):
                    if current in seen or current > 10**12:
                        break
                    seen.add(current)
                    for op_name in hyp.pipeline.operations:
                        if op_name in all_ops:
                            current = all_ops[op_name].apply(current)
                endpoints[current] += 1
            
            if endpoints:
                top_val, top_count = endpoints.most_common(1)[0]
                convergence = top_count / sample_size
                results.append((top_val, convergence))
        
        hyp.tested = True
        
        # Evaluate results
        if results:
            avg_convergence = np.mean([r[1] for r in results])
            
            if hyp.expected_pattern == 'constant' and avg_convergence > 0.9:
                hyp.confirmed = True
                hyp.confidence = avg_convergence
                self.confirmed.append(hyp)
                return True
            elif hyp.expected_pattern == 'cycle' and avg_convergence > 0.5:
                hyp.confirmed = True
                hyp.confidence = avg_convergence
                self.confirmed.append(hyp)
                return True
        
        self.refuted.append(hyp)
        return False


# =============================================================================
# AUTONOMOUS RESEARCHER
# =============================================================================

@dataclass
class ResearchConfig:
    """Configuration for autonomous research."""
    max_iterations: int = 10
    population_size: int = 40
    hypotheses_per_iteration: int = 10
    sample_size: int = 3000
    db_path: str = "autonomous_research.db"
    
    # Self-improvement
    learning_rate: float = 0.1
    exploration_decay: float = 0.95


class AutonomousResearcher:
    """
    Fully autonomous research system.

    Cycle:
    1. Generate hypotheses
    2. Test hypotheses
    3. Analyze results
    4. Generate new operations/combinations
    5. Repeat
    """
    
    def __init__(self, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self.op_factory = DynamicOperationFactory()
        self.hypothesis_engine = HypothesisEngine(self.op_factory)
        self.scorer = ImprovedScorer()
        self.rng = random.Random(42)
        
        self.discoveries: List[DiscoveredPattern] = []
        self.iteration = 0
        
        # Learning state
        self.op_success_rate: Dict[str, float] = defaultdict(lambda: 0.5)
        self.combo_success_rate: Dict[str, float] = defaultdict(lambda: 0.5)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS research_discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                pipeline TEXT,
                pattern_type TEXT,
                attractor_value INTEGER,
                cycle_values TEXT,
                score REAL,
                hypothesis_origin TEXT,
                timestamp REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS research_hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                description TEXT,
                pipeline TEXT,
                confirmed INTEGER,
                confidence REAL,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def _save_discovery(self, pattern: DiscoveredPattern, hypothesis_origin: str = None):
        """Save discovery."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO research_discoveries 
            (iteration, pipeline, pattern_type, attractor_value, cycle_values, score, hypothesis_origin, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.iteration,
            pattern.pipeline.signature(),
            pattern.pattern_type.name,
            pattern.attractor_value,
            json.dumps(pattern.cycle_values) if pattern.cycle_values else None,
            pattern.interestingness_score,
            hypothesis_origin,
            time.time(),
        ))
        conn.commit()
        conn.close()
    
    def _save_hypothesis(self, hyp: Hypothesis):
        """Save hypothesis."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO research_hypotheses 
            (iteration, description, pipeline, confirmed, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.iteration,
            hyp.description,
            hyp.pipeline.signature(),
            1 if hyp.confirmed else 0,
            hyp.confidence,
            time.time(),
        ))
        conn.commit()
        conn.close()
    
    def explore_random(self, count: int = 20) -> List[DiscoveredPattern]:
        """Random exploration of new pipelines."""
        all_ops = self.op_factory.get_all_ops()
        op_names = list(all_ops.keys())
        
        discoveries = []
        
        for i in range(count):
            if i % 5 == 0:
                print(f"      Exploring {i}/{count}...", flush=True)
            # Select operations weighted by success rate
            length = self.rng.randint(1, 3)
            ops = []
            for _ in range(length):
                # Thompson-like selection
                weights = [self.op_success_rate[op] for op in op_names]
                total = sum(weights)
                weights = [w/total for w in weights]
                op = self.rng.choices(op_names, weights=weights)[0]
                ops.append(op)
            
            try:
                pipeline = Pipeline(ops)
                pattern = self._evaluate_pipeline(pipeline)
                
                if pattern and pattern.interestingness_score > 30:
                    discoveries.append(pattern)
                    self._update_success_rates(ops, True)
                else:
                    self._update_success_rates(ops, False)
            except:
                pass
        
        return discoveries
    
    def _evaluate_pipeline(self, pipeline: Pipeline) -> Optional[DiscoveredPattern]:
        """Evaluate a pipeline."""
        all_ops = self.op_factory.get_all_ops()
        
        best_pattern = None
        best_score = 0.0
        
        for digit_range in [(3,3), (4,4), (5,5)]:
            min_n, max_n = 10**(digit_range[0]-1), 10**digit_range[1] - 1
            numbers = np.random.randint(min_n, max_n + 1, size=self.config.sample_size)
            
            endpoints = Counter()
            cycles = Counter()
            
            for n in numbers:
                current = int(n)
                sequence = [current]
                seen = {current: 0}
                
                for step in range(100):
                    for op_name in pipeline.operations:
                        if op_name in all_ops:
                            current = all_ops[op_name].apply(current)
                        if current > 10**12:
                            break
                    
                    if current in seen:
                        cycle_start = seen[current]
                        cycle = tuple(sequence[cycle_start:])
                        cycles[cycle] += 1
                        break
                    
                    seen[current] = len(sequence)
                    sequence.append(current)
                else:
                    endpoints[current] += 1
            
            # Determine pattern
            total = len(numbers)
            
            if cycles:
                top_cycle, count = cycles.most_common(1)[0]
                rate = count / total
                if rate > 0.5:
                    pattern = DiscoveredPattern(
                        pipeline=pipeline,
                        pattern_type=PatternType.CYCLE,
                        cycle_values=list(top_cycle),
                        convergence_rate=rate,
                        tested_range=(min_n, max_n),
                        num_tested=total,
                    )
                    score = self.scorer.score(pattern)
                    pattern.interestingness_score = score
                    
                    if score > best_score:
                        best_score = score
                        best_pattern = pattern
            
            if endpoints:
                top_val, count = endpoints.most_common(1)[0]
                rate = count / total
                if rate > 0.9:
                    pattern = DiscoveredPattern(
                        pipeline=pipeline,
                        pattern_type=PatternType.UNIVERSAL_CONSTANT,
                        attractor_value=top_val,
                        convergence_rate=rate,
                        tested_range=(min_n, max_n),
                        num_tested=total,
                    )
                    score = self.scorer.score(pattern)
                    pattern.interestingness_score = score
                    
                    if score > best_score:
                        best_score = score
                        best_pattern = pattern
        
        return best_pattern
    
    def _update_success_rates(self, ops: List[str], success: bool):
        """Update success rates for operations."""
        for op in ops:
            old_rate = self.op_success_rate[op]
            if success:
                self.op_success_rate[op] = old_rate + self.config.learning_rate * (1 - old_rate)
            else:
                self.op_success_rate[op] = old_rate - self.config.learning_rate * old_rate * 0.5
    
    def generate_new_operations(self):
        """Generate new operations based on patterns."""
        # Generate power-sum variations
        for p in range(2, 8):
            self.op_factory.generate_power_sum(p)
        
        # Generate modulo variations
        for m in [7, 9, 11, 13, 17, 19]:
            self.op_factory.generate_modulo(m)
        
        # Generate weighted sums
        self.op_factory.generate_weighted_sum([1, 2, 1])
        self.op_factory.generate_weighted_sum([1, -1, 1, -1])
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run one research iteration."""
        import sys
        print(f"\n{'='*60}", flush=True)
        print(f"🔬 ITERATION {self.iteration}", flush=True)
        print(f"{'='*60}", flush=True)
        sys.stdout.flush()
        
        results = {
            'iteration': self.iteration,
            'new_discoveries': 0,
            'hypotheses_tested': 0,
            'hypotheses_confirmed': 0,
        }
        
        # Step 1: Random exploration
        print("\n📊 Phase 1: Random exploration...", flush=True)
        random_discoveries = self.explore_random(self.config.population_size)
        
        for disc in random_discoveries:
            if disc.pipeline.signature() not in [d.pipeline.signature() for d in self.discoveries]:
                self.discoveries.append(disc)
                self._save_discovery(disc, "random_exploration")
                results['new_discoveries'] += 1
        
        print(f"   New discoveries: {results['new_discoveries']}", flush=True)

        # Step 2: Generate hypotheses
        print("\n🧠 Phase 2: Hypothesis generation...", flush=True)
        new_hypotheses = self.hypothesis_engine.generate_hypotheses(self.discoveries)
        print(f"   New hypotheses: {len(new_hypotheses)}", flush=True)

        # Step 3: Test hypotheses
        print("\n🧪 Phase 3: Hypothesis testing...", flush=True)
        for hyp in new_hypotheses[:self.config.hypotheses_per_iteration]:
            confirmed = self.hypothesis_engine.test_hypothesis(hyp)
            self._save_hypothesis(hyp)
            results['hypotheses_tested'] += 1
            
            if confirmed:
                results['hypotheses_confirmed'] += 1
                print(f"   ✅ CONFIRMED: {hyp.description[:50]}...", flush=True)

                # Add to discoveries
                pattern = self._evaluate_pipeline(hyp.pipeline)
                if pattern:
                    self.discoveries.append(pattern)
                    self._save_discovery(pattern, hyp.description)
        
        # Step 4: Generate new operations
        print("\n🔧 Phase 4: Operation generation...", flush=True)
        self.generate_new_operations()
        print(f"   Total operations: {len(self.op_factory.get_all_ops())}", flush=True)

        # Step 5: Report top discoveries
        print("\n🏆 Top discoveries this iteration:")
        top = sorted(self.discoveries, key=lambda x: x.interestingness_score, reverse=True)[:5]
        for i, d in enumerate(top, 1):
            trivial = self.scorer.is_trivial(d)
            if not trivial:
                print(f"   {i}. {d.pipeline.signature()} (score: {d.interestingness_score:.1f})")
                if d.attractor_value:
                    print(f"      Attractor: {d.attractor_value}")
                if d.cycle_values:
                    print(f"      Cycle: {d.cycle_values[:3]}...")
        
        self.iteration += 1
        return results
    
    def run(self, verbose: bool = True):
        """Run full research."""
        print("█" * 70)
        print("  SYNTRIAD AUTONOMOUS RESEARCHER v3.0")
        print("█" * 70)
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Population: {self.config.population_size}")
        print("█" * 70)
        
        start_time = time.time()
        all_results = []
        
        for _ in range(self.config.max_iterations):
            results = self.run_iteration()
            all_results.append(results)
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "█" * 70)
        print("  RESEARCH COMPLETE")
        print("█" * 70)
        print(f"Total time: {total_time:.1f}s")
        print(f"Total discoveries: {len(self.discoveries)}")
        print(f"Confirmed hypotheses: {len(self.hypothesis_engine.confirmed)}")

        # Top 10 non-trivial discoveries
        print("\n🏆 TOP 10 NON-TRIVIAL DISCOVERIES:")
        non_trivial = [d for d in self.discoveries if not self.scorer.is_trivial(d)]
        top = sorted(non_trivial, key=lambda x: x.interestingness_score, reverse=True)[:10]
        
        for i, d in enumerate(top, 1):
            print(f"\n{i:2d}. {d.pipeline.signature()}")
            print(f"    Score: {d.interestingness_score:.1f} | Type: {d.pattern_type.name}")
            if d.attractor_value:
                print(f"    Attractor: {d.attractor_value}")
            if d.cycle_values:
                cycle_str = str(d.cycle_values[:5])
                if len(d.cycle_values) > 5:
                    cycle_str = cycle_str[:-1] + ", ...]"
                print(f"    Cycle: {cycle_str}")
        
        # Most successful operations
        print("\n📊 MOST SUCCESSFUL OPERATIONS:")
        sorted_ops = sorted(self.op_success_rate.items(), key=lambda x: -x[1])[:10]
        for op, rate in sorted_ops:
            print(f"   {op:25s}: {rate:.3f}")
        
        return self.discoveries


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTRIAD Autonomous Researcher')
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--population', type=int, default=30)
    parser.add_argument('--db', type=str, default='autonomous_research.db')
    args = parser.parse_args()
    
    config = ResearchConfig(
        max_iterations=args.iterations,
        population_size=args.population,
        db_path=args.db,
    )
    
    researcher = AutonomousResearcher(config)
    discoveries = researcher.run()
    
    print(f"\n✅ Research complete! {len(discoveries)} patterns discovered.")


if __name__ == "__main__":
    main()
