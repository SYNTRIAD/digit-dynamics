#!/usr/bin/env python3
"""
SYNTRIAD Symmetry Discovery - Unified Launcher
===============================================

Start verschillende discovery modes:
- explore: CPU evolutionaire zoektocht
- gpu: GPU-versnelde batch experimenten  
- meta: Zelf-lerend meta-systeem
- demo: Quick demo van alle capabilities

Usage:
    python run_discovery.py --mode demo
    python run_discovery.py --mode explore --generations 100
    python run_discovery.py --mode gpu --experiments kaprekar,1089,happy
"""

import argparse
import sys


def run_demo(args):
    """Quick demo van capabilities."""
    print("█" * 70)
    print("  SYNTRIAD SYMMETRY DISCOVERY - DEMO")
    print("█" * 70)
    
    from symmetry_discovery_engine import Pipeline, OPERATION_REGISTRY
    
    # Demo 1: Bekende patronen
    print("\n\n" + "="*60)
    print("DEMO 1: BEKENDE PATRONEN VERIFIËREN")
    print("="*60)
    
    known_pipelines = [
        (['kaprekar_step'], "Kaprekar's constante"),
        (['truc_1089'], "1089-truc"),
        (['happy_step'], "Happy numbers"),
        (['digitsum'], "Digital root"),
        (['sub_reverse', 'add_reverse'], "Reverse combo"),
    ]
    
    for ops, name in known_pipelines:
        pipeline = Pipeline(ops)
        print(f"\n{name}: {pipeline.signature()}")
        
        for n in [752, 6174, 12345, 9876]:
            final, seq = pipeline.apply(n, max_iter=50)
            seq_str = ' → '.join(map(str, seq[:8]))
            if len(seq) > 8:
                seq_str += ' → ...'
            print(f"  {n}: {seq_str} → {final}")
    
    # Demo 2: Random pipeline
    print("\n\n" + "="*60)
    print("DEMO 2: RANDOM PIPELINE GENERATIE")
    print("="*60)
    
    import random
    rng = random.Random(42)
    
    for _ in range(5):
        pipeline = Pipeline.random(rng, max_length=3)
        print(f"\n{pipeline.signature()}")
        
        for n in [123, 4567, 98765]:
            final, seq = pipeline.apply(n, max_iter=20)
            print(f"  {n} → {final} (in {len(seq)-1} stappen)")
    
    # Demo 3: Operaties
    print("\n\n" + "="*60)
    print("DEMO 3: BESCHIKBARE OPERATIES")
    print("="*60)
    
    for name, op in sorted(OPERATION_REGISTRY.items()):
        example = op.apply(12345)
        print(f"  {name:20s}: 12345 → {example}")
    
    print("\n✅ Demo compleet!")


def run_explore(args):
    """CPU evolutionaire zoektocht."""
    from symmetry_discovery_engine import DiscoveryEngine, DiscoveryConfig
    
    config = DiscoveryConfig(
        num_cpu_workers=args.workers,
        population_size=args.population,
        db_path=args.db,
    )
    
    engine = DiscoveryEngine(config)
    discoveries = engine.run(num_generations=args.generations)
    return discoveries


def run_gpu(args):
    """GPU-versnelde experimenten."""
    try:
        from gpu_symmetry_hunter import GPUSymmetryHunter, GPUExperimentConfig
    except ImportError as e:
        print(f"❌ GPU module niet beschikbaar: {e}")
        return []
    
    config = GPUExperimentConfig(batch_size=args.batch_size, max_iterations=200)
    hunter = GPUSymmetryHunter(config)
    
    experiments = args.experiments.split(',') if args.experiments else ['kaprekar', '1089', 'happy']
    results = []
    
    if 'kaprekar' in experiments:
        for digits in [3, 4, 5, 6]:
            r = hunter.run_kaprekar_experiment((digits, digits), count=1_000_000)
            results.append(r)
    
    if '1089' in experiments:
        for digits in [3, 4, 5, 6]:
            r = hunter.run_1089_experiment((digits, digits), constraint="descending")
            results.append(r)
    
    if 'happy' in experiments:
        r = hunter.run_happy_number_experiment((1, 10_000_000), count=2_000_000)
        results.append(r)
    
    if 'palindrome' in experiments:
        r = hunter.run_palindrome_experiment((2, 2), count=1_000_000, max_iter=500)
        results.append(r)
    
    if 'discover' in experiments:
        discoveries = hunter.discover_new_constants()
        results.extend(discoveries)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='SYNTRIAD Symmetry Discovery System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_discovery.py --mode demo
  python run_discovery.py --mode explore --generations 100
  python run_discovery.py --mode gpu --experiments kaprekar,1089
        """
    )
    
    parser.add_argument('--mode', choices=['explore', 'gpu', 'demo'], default='demo')
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--batch-size', type=int, default=1_000_000)
    parser.add_argument('--db', type=str, default='discoveries.db')
    parser.add_argument('--experiments', type=str, default='kaprekar,1089,happy')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'explore':
        run_explore(args)
    elif args.mode == 'gpu':
        run_gpu(args)


if __name__ == "__main__":
    main()
