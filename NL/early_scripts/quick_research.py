#!/usr/bin/env python3
"""
SYNTRIAD Quick Research - Snelle iteratieve ontdekking met live output
"""

import numpy as np
import random
import time
import sys
from collections import Counter
from meta_discovery_engine import Pipeline, EXTENDED_OPERATIONS, ImprovedScorer, PatternType, DiscoveredPattern

def log(msg):
    print(msg, flush=True)

def evaluate_pipeline(pipeline, sample_size=500):
    """Snelle pipeline evaluatie met overflow bescherming."""
    endpoints = Counter()
    cycles = Counter()
    
    # Skip problematische operaties
    skip_ops = {'collatz_step', 'pair_sum', 'mul_reverse', 'add_reverse'}
    if any(op in skip_ops for op in pipeline.operations):
        return None
    
    for digit_len in [3, 4]:
        low, high = 10**(digit_len-1), 10**digit_len - 1
        numbers = np.random.randint(low, high+1, size=sample_size)
        
        for n in numbers:
            current = int(n)
            seen = {current: 0}
            seq = [current]
            
            try:
                for step in range(30):  # Minder stappen
                    for op_name in pipeline.operations:
                        current = EXTENDED_OPERATIONS[op_name].apply(current)
                        if current > 10**9 or current < -10**9:  # Striktere limiet
                            raise OverflowError()
                    
                    if current in seen:
                        cycle = tuple(seq[seen[current]:])
                        cycles[cycle] += 1
                        break
                    seen[current] = len(seq)
                    seq.append(current)
                else:
                    endpoints[current] += 1
            except:
                pass  # Skip problematische getallen
    
    total = sample_size * 2
    
    if cycles:
        top_cycle, count = cycles.most_common(1)[0]
        if count / total > 0.3:
            return {
                'type': 'CYCLE',
                'value': list(top_cycle)[:5],
                'rate': count / total,
            }
    
    if endpoints:
        top_val, count = endpoints.most_common(1)[0]
        if count / total > 0.5:
            return {
                'type': 'CONSTANT',
                'value': top_val,
                'rate': count / total,
            }
    
    return None

def main():
    log("█" * 60)
    log("  SYNTRIAD QUICK RESEARCH")
    log("█" * 60)
    
    scorer = ImprovedScorer()
    rng = random.Random(42)
    op_names = list(EXTENDED_OPERATIONS.keys())
    
    discoveries = []
    seen_sigs = set()
    
    # Succes tracking
    op_scores = {op: 1.0 for op in op_names}
    
    for iteration in range(5):
        log(f"\n{'='*50}")
        log(f"🔬 ITERATIE {iteration}")
        log(f"{'='*50}")
        
        iter_discoveries = 0
        
        # Genereer en test pipelines
        for i in range(20):
            # Gewogen selectie
            weights = [op_scores[op] for op in op_names]
            total_w = sum(weights)
            weights = [w/total_w for w in weights]
            
            length = rng.randint(1, 3)
            ops = rng.choices(op_names, weights=weights, k=length)
            
            try:
                pipeline = Pipeline(ops)
                sig = pipeline.signature()
                
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                
                result = evaluate_pipeline(pipeline)
                
                if result:
                    # Check trivialiteit
                    is_trivial = False
                    if result['type'] == 'CONSTANT' and result['value'] in {0, 1, 2}:
                        is_trivial = True
                    if result['type'] == 'CYCLE':
                        if all(v in {0, 1, 2} for v in result['value']):
                            is_trivial = True
                    
                    if not is_trivial and result['rate'] > 0.5:
                        discoveries.append({
                            'pipeline': sig,
                            'result': result,
                        })
                        iter_discoveries += 1
                        
                        # Update scores
                        for op in ops:
                            op_scores[op] *= 1.2
                        
                        log(f"  🎯 {sig}")
                        log(f"     {result['type']}: {result['value']} ({result['rate']*100:.0f}%)")
                else:
                    for op in ops:
                        op_scores[op] *= 0.95
                        
            except Exception as e:
                pass
            
            if i % 5 == 0:
                log(f"  ... tested {i+1}/20")
        
        log(f"\n  Nieuwe ontdekkingen: {iter_discoveries}")
        log(f"  Totaal: {len(discoveries)}")
        
        # Top operaties
        top_ops = sorted(op_scores.items(), key=lambda x: -x[1])[:5]
        log(f"  Top ops: {[op for op, _ in top_ops]}")
    
    # Finale samenvatting
    log("\n" + "█" * 60)
    log("  RESULTATEN")
    log("█" * 60)
    log(f"Totaal ontdekkingen: {len(discoveries)}")
    
    # Sorteer op convergentie rate
    discoveries.sort(key=lambda x: -x['result']['rate'])
    
    log("\n🏆 TOP ONTDEKKINGEN:")
    for i, d in enumerate(discoveries[:15], 1):
        log(f"{i:2d}. {d['pipeline']}")
        log(f"    {d['result']['type']}: {d['result']['value']}")

if __name__ == "__main__":
    main()
