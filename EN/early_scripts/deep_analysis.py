#!/usr/bin/env python3
"""
SYNTRIAD Deep Analysis - Investigate the most interesting discoveries
======================================================================

Analyzes the top discoveries and searches for mathematical patterns.
"""

import sqlite3
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from meta_discovery_engine import Pipeline, EXTENDED_OPERATIONS

def analyze_cycle(pipeline: Pipeline, start_range: Tuple[int, int], sample_size: int = 10000) -> Dict:
    """Analyze a pipeline and find all cycles."""
    min_n, max_n = start_range
    numbers = np.random.randint(min_n, max_n + 1, size=sample_size)
    
    cycle_counter = Counter()
    attractor_counter = Counter()
    steps_to_cycle = []
    
    for n in numbers:
        n = int(n)
        sequence = [n]
        seen = {n: 0}
        current = n
        
        for step in range(200):
            for op in [EXTENDED_OPERATIONS[name] for name in pipeline.operations]:
                current = op.apply(current)
                if current > 10**15:
                    break
            
            if current in seen:
                cycle_start = seen[current]
                cycle = tuple(sequence[cycle_start:])
                cycle_counter[cycle] += 1
                steps_to_cycle.append(cycle_start)
                break
            
            seen[current] = len(sequence)
            sequence.append(current)
    
    return {
        'cycles': cycle_counter.most_common(10),
        'avg_steps': np.mean(steps_to_cycle) if steps_to_cycle else 0,
        'unique_cycles': len(cycle_counter),
    }


def find_fixed_points(op_name: str, max_n: int = 1000000) -> List[int]:
    """Find fixed points for an operation."""
    op = EXTENDED_OPERATIONS[op_name]
    fixed_points = []
    
    for n in range(1, min(max_n, 100000)):
        if op.apply(n) == n:
            fixed_points.append(n)
    
    return fixed_points


def analyze_factorion_cycle():
    """Analyze the famous factorion cycle [169, 363601, 1454]."""
    print("\n" + "=" * 70)
    print("🔬 FACTORION CYCLE ANALYSIS")
    print("=" * 70)
    
    op = EXTENDED_OPERATIONS['digit_factorial_sum']
    
    # The cycle
    cycle = [169, 363601, 1454]
    print(f"\nCycle: {cycle}")
    
    for n in cycle:
        result = op.apply(n)
        digits = [int(d) for d in str(n)]
        factorials = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        terms = [f"{d}!={factorials[d]}" for d in digits]
        print(f"  {n} = {' + '.join(terms)} = {result}")
    
    # Find all numbers that converge to this cycle
    print("\n📊 Convergence to factorion cycle:")
    converge_count = 0
    other_cycles = Counter()
    
    for n in range(1, 100000):
        current = n
        seen = set()
        for _ in range(100):
            if current in seen:
                break
            seen.add(current)
            current = op.apply(current)
        
        if current in {169, 363601, 1454}:
            converge_count += 1
        else:
            other_cycles[current] += 1
    
    print(f"  Numbers 1-99999 converging to [169, 363601, 1454]: {converge_count}")
    print(f"  Other endpoints: {other_cycles.most_common(5)}")


def analyze_narcissistic_numbers():
    """Analyze narcissistic numbers (Armstrong numbers)."""
    print("\n" + "=" * 70)
    print("🔬 NARCISSISTIC NUMBERS (digit_pow_3)")
    print("=" * 70)
    
    op = EXTENDED_OPERATIONS['digit_pow_3']
    
    # Find all narcissistic numbers up to 1M
    narcissistic = []
    for n in range(1, 1000000):
        if op.apply(n) == n:
            narcissistic.append(n)
    
    print(f"\nNarcissistic numbers (3-digit): {[n for n in narcissistic if 100 <= n <= 999]}")
    print(f"All found: {narcissistic}")
    
    # Analyze convergence
    print("\n📊 Where do numbers converge to?")
    endpoints = Counter()
    for n in range(100, 1000):
        current = n
        for _ in range(50):
            new = op.apply(current)
            if new == current:
                break
            current = new
        endpoints[current] += 1
    
    for endpoint, count in endpoints.most_common(10):
        pct = 100 * count / 900
        print(f"  {endpoint}: {count} numbers ({pct:.1f}%)")


def analyze_kaprekar_variants():
    """Analyze variants of Kaprekar's constant."""
    print("\n" + "=" * 70)
    print("🔬 KAPREKAR VARIANTS BY DIGIT COUNT")
    print("=" * 70)
    
    op = EXTENDED_OPERATIONS['kaprekar_step']
    
    for num_digits in range(2, 8):
        low = 10 ** (num_digits - 1)
        high = 10 ** num_digits - 1
        
        endpoints = Counter()
        for n in range(low, min(high + 1, low + 50000)):
            current = n
            for _ in range(100):
                new = op.apply(current)
                if new == current or new == 0:
                    break
                current = new
            endpoints[current] += 1
        
        print(f"\n{num_digits}-digit:")
        for endpoint, count in endpoints.most_common(5):
            print(f"  {endpoint}: {count}")


def analyze_new_discovery(pipeline_str: str, digit_range: Tuple[int, int] = (3, 6)):
    """Analyze a specific pipeline discovery."""
    ops = pipeline_str.split(" → ")
    pipeline = Pipeline(ops)
    
    print(f"\n" + "=" * 70)
    print(f"🔬 ANALYSE: {pipeline_str}")
    print("=" * 70)
    
    for num_digits in range(digit_range[0], digit_range[1] + 1):
        low = 10 ** (num_digits - 1)
        high = 10 ** num_digits - 1
        
        result = analyze_cycle(pipeline, (low, high), sample_size=5000)
        
        print(f"\n{num_digits}-digit:")
        print(f"  Unique cycles: {result['unique_cycles']}")
        print(f"  Avg. steps to cycle: {result['avg_steps']:.1f}")
        
        for cycle, count in result['cycles'][:3]:
            pct = 100 * count / 5000
            cycle_str = str(list(cycle)[:5])
            if len(cycle) > 5:
                cycle_str = cycle_str[:-1] + ", ...]"
            print(f"  Cycle {cycle_str}: {count} ({pct:.1f}%)")


def discover_new_patterns():
    """Search for new, undiscovered patterns."""
    print("\n" + "=" * 70)
    print("🔬 SEARCHING FOR NEW PATTERNS")
    print("=" * 70)
    
    # Interesting combinations to test
    interesting_combos = [
        ['digit_pow_3', 'kaprekar_step'],
        ['complement_9', 'kaprekar_step'],
        ['swap_ends', 'truc_1089'],
        ['digit_pow_4', 'happy_step'],
        ['persistence', 'digit_factorial_sum'],
        ['sq_diff', 'kaprekar_step'],
        ['alt_prod', 'digit_factorial_sum'],
        ['complement_9', 'digit_pow_3'],
        ['mod_9', 'digit_factorial_sum'],
        ['cycle_2', 'kaprekar_step'],
    ]
    
    results = []
    
    for ops in interesting_combos:
        try:
            pipeline = Pipeline(ops)
            
            # Test on 4-digit numbers
            endpoints = Counter()
            for n in range(1000, 10000):
                current = n
                seen = set()
                for _ in range(100):
                    if current in seen or current > 10**12:
                        break
                    seen.add(current)
                    for op in [EXTENDED_OPERATIONS[name] for name in ops]:
                        current = op.apply(current)
                
                endpoints[current] += 1
            
            top_endpoint, top_count = endpoints.most_common(1)[0]
            convergence = 100 * top_count / 9000
            
            if convergence > 50 and top_endpoint not in {0, 1, 2}:
                results.append({
                    'pipeline': pipeline.signature(),
                    'attractor': top_endpoint,
                    'convergence': convergence,
                })
                print(f"\n🎯 {pipeline.signature()}")
                print(f"   Attractor: {top_endpoint} ({convergence:.1f}%)")
        except Exception as e:
            pass
    
    return results


def main():
    print("█" * 70)
    print("  SYNTRIAD DEEP ANALYSIS")
    print("█" * 70)
    
    # 1. Factorion cycle
    analyze_factorion_cycle()
    
    # 2. Narcissistic numbers
    analyze_narcissistic_numbers()
    
    # 3. Kaprekar variants
    analyze_kaprekar_variants()
    
    # 4. Analyze top discoveries
    top_discoveries = [
        "digit_pow_4 → truc_1089",
        "sort_diff → swap_ends",
        "truc_1089 → digit_pow_4",
    ]
    
    for discovery in top_discoveries:
        analyze_new_discovery(discovery)
    
    # 5. Search for new patterns
    new_patterns = discover_new_patterns()
    
    print("\n" + "█" * 70)
    print("  ANALYSIS COMPLETE")
    print("█" * 70)


if __name__ == "__main__":
    main()
