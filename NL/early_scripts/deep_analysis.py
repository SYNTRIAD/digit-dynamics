#!/usr/bin/env python3
"""
SYNTRIAD Deep Analysis - Onderzoek de meest interessante ontdekkingen
======================================================================

Analyseert de top ontdekkingen en zoekt naar wiskundige patronen.
"""

import sqlite3
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set
from meta_discovery_engine import Pipeline, EXTENDED_OPERATIONS

def analyze_cycle(pipeline: Pipeline, start_range: Tuple[int, int], sample_size: int = 10000) -> Dict:
    """Analyseer een pipeline en vind alle cycli."""
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
    """Vind vaste punten voor een operatie."""
    op = EXTENDED_OPERATIONS[op_name]
    fixed_points = []
    
    for n in range(1, min(max_n, 100000)):
        if op.apply(n) == n:
            fixed_points.append(n)
    
    return fixed_points


def analyze_factorion_cycle():
    """Analyseer de beroemde factorion cyclus [169, 363601, 1454]."""
    print("\n" + "=" * 70)
    print("🔬 FACTORION CYCLUS ANALYSE")
    print("=" * 70)
    
    op = EXTENDED_OPERATIONS['digit_factorial_sum']
    
    # De cyclus
    cycle = [169, 363601, 1454]
    print(f"\nCyclus: {cycle}")
    
    for n in cycle:
        result = op.apply(n)
        digits = [int(d) for d in str(n)]
        factorials = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        terms = [f"{d}!={factorials[d]}" for d in digits]
        print(f"  {n} = {' + '.join(terms)} = {result}")
    
    # Vind alle getallen die naar deze cyclus convergeren
    print("\n📊 Convergentie naar factorion cyclus:")
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
    
    print(f"  Getallen 1-99999 die naar [169, 363601, 1454] convergeren: {converge_count}")
    print(f"  Andere eindpunten: {other_cycles.most_common(5)}")


def analyze_narcissistic_numbers():
    """Analyseer narcissistische getallen (Armstrong numbers)."""
    print("\n" + "=" * 70)
    print("🔬 NARCISSISTISCHE GETALLEN (digit_pow_3)")
    print("=" * 70)
    
    op = EXTENDED_OPERATIONS['digit_pow_3']
    
    # Vind alle narcissistische getallen tot 1M
    narcissistic = []
    for n in range(1, 1000000):
        if op.apply(n) == n:
            narcissistic.append(n)
    
    print(f"\nNarcissistische getallen (3-cijferig): {[n for n in narcissistic if 100 <= n <= 999]}")
    print(f"Alle gevonden: {narcissistic}")
    
    # Analyseer convergentie
    print("\n📊 Waar convergeren getallen naar?")
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
        print(f"  {endpoint}: {count} getallen ({pct:.1f}%)")


def analyze_kaprekar_variants():
    """Analyseer varianten van Kaprekar's constante."""
    print("\n" + "=" * 70)
    print("🔬 KAPREKAR VARIANTEN PER CIJFERAANTAL")
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
        
        print(f"\n{num_digits}-cijferig:")
        for endpoint, count in endpoints.most_common(5):
            print(f"  {endpoint}: {count}")


def analyze_new_discovery(pipeline_str: str, digit_range: Tuple[int, int] = (3, 6)):
    """Analyseer een specifieke pipeline ontdekking."""
    ops = pipeline_str.split(" → ")
    pipeline = Pipeline(ops)
    
    print(f"\n" + "=" * 70)
    print(f"🔬 ANALYSE: {pipeline_str}")
    print("=" * 70)
    
    for num_digits in range(digit_range[0], digit_range[1] + 1):
        low = 10 ** (num_digits - 1)
        high = 10 ** num_digits - 1
        
        result = analyze_cycle(pipeline, (low, high), sample_size=5000)
        
        print(f"\n{num_digits}-cijferig:")
        print(f"  Unieke cycli: {result['unique_cycles']}")
        print(f"  Gem. stappen tot cyclus: {result['avg_steps']:.1f}")
        
        for cycle, count in result['cycles'][:3]:
            pct = 100 * count / 5000
            cycle_str = str(list(cycle)[:5])
            if len(cycle) > 5:
                cycle_str = cycle_str[:-1] + ", ...]"
            print(f"  Cyclus {cycle_str}: {count} ({pct:.1f}%)")


def discover_new_patterns():
    """Zoek naar nieuwe, onontdekte patronen."""
    print("\n" + "=" * 70)
    print("🔬 ZOEKEN NAAR NIEUWE PATRONEN")
    print("=" * 70)
    
    # Interessante combinaties om te testen
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
            
            # Test op 4-cijferige getallen
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
    
    # 1. Factorion cyclus
    analyze_factorion_cycle()
    
    # 2. Narcissistische getallen
    analyze_narcissistic_numbers()
    
    # 3. Kaprekar varianten
    analyze_kaprekar_variants()
    
    # 4. Analyseer top ontdekkingen
    top_discoveries = [
        "digit_pow_4 → truc_1089",
        "sort_diff → swap_ends",
        "truc_1089 → digit_pow_4",
    ]
    
    for discovery in top_discoveries:
        analyze_new_discovery(discovery)
    
    # 5. Zoek nieuwe patronen
    new_patterns = discover_new_patterns()
    
    print("\n" + "█" * 70)
    print("  ANALYSE COMPLEET")
    print("█" * 70)


if __name__ == "__main__":
    main()
