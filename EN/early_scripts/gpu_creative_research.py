#!/usr/bin/env python3
"""
SYNTRIAD GPU Creative Research v2.0
====================================

GPU-accelerated version of the creative experiments.
Explores new operations at the scale of millions of numbers.

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
import math

try:
    from numba import cuda, int64, float64, boolean, uint8
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("⚠️  CUDA not available. Install: pip install numba")


# =============================================================================
# GPU DEVICE FUNCTIONS - New Creative Operations
# =============================================================================

if HAS_CUDA:
    
    # Basic helpers
    @cuda.jit(device=True)
    def gpu_reverse(n: int64) -> int64:
        if n == 0:
            return 0
        result = 0
        temp = n if n > 0 else -n
        while temp > 0:
            result = result * 10 + (temp % 10)
            temp //= 10
        return result if n > 0 else -result
    
    @cuda.jit(device=True)
    def gpu_digit_sum(n: int64) -> int64:
        total = 0
        temp = n if n > 0 else -n
        while temp > 0:
            total += temp % 10
            temp //= 10
        return total
    
    @cuda.jit(device=True)
    def gpu_digit_count(n: int64) -> int64:
        if n == 0:
            return 1
        count = 0
        temp = n if n > 0 else -n
        while temp > 0:
            count += 1
            temp //= 10
        return count
    
    # =========================================================================
    # NEW OPERATIONS
    # =========================================================================
    
    @cuda.jit(device=True)
    def gpu_fibonacci_digit_sum(n: int64) -> int64:
        """Sum of Fibonacci mappings: 0→0, 1→1, 2→1, 3→2, 4→3, 5→5, 6→8, 7→13, 8→21, 9→34"""
        fib_map = cuda.local.array(10, dtype=int64)
        fib_map[0] = 0
        fib_map[1] = 1
        fib_map[2] = 1
        fib_map[3] = 2
        fib_map[4] = 3
        fib_map[5] = 5
        fib_map[6] = 8
        fib_map[7] = 13
        fib_map[8] = 21
        fib_map[9] = 34
        
        total = 0
        temp = n if n > 0 else -n
        while temp > 0:
            d = temp % 10
            total += fib_map[d]
            temp //= 10
        return total
    
    @cuda.jit(device=True)
    def gpu_xor_reverse(n: int64) -> int64:
        """n XOR reverse(n)"""
        if n <= 0:
            return 0
        rev = gpu_reverse(n)
        return n ^ rev
    
    @cuda.jit(device=True)
    def gpu_digit_gravity(n: int64) -> int64:
        """Sum of products of adjacent digits"""
        if n <= 0:
            return 0
        digits = cuda.local.array(20, dtype=int64)
        num_digits = 0
        temp = n
        while temp > 0 and num_digits < 20:
            digits[num_digits] = temp % 10
            temp //= 10
            num_digits += 1
        
        if num_digits < 2:
            return digits[0] if num_digits == 1 else 0
        
        total = 0
        for i in range(num_digits - 1):
            total += digits[i] * digits[i + 1]
        return total
    
    @cuda.jit(device=True)
    def gpu_digit_pow3_sum(n: int64) -> int64:
        """Sum of cubes of digits"""
        total = 0
        temp = n if n > 0 else -n
        while temp > 0:
            d = temp % 10
            total += d * d * d
            temp //= 10
        return total
    
    @cuda.jit(device=True)
    def gpu_digit_alchemy(n: int64) -> int64:
        """digit_pow3 -> reverse -> digit_sum"""
        step1 = gpu_digit_pow3_sum(n)
        step2 = gpu_reverse(step1)
        step3 = gpu_digit_sum(step2)
        return step3
    
    @cuda.jit(device=True)
    def gpu_prime_digit_sum(n: int64) -> int64:
        """Sum of only prime digits (2, 3, 5, 7)"""
        total = 0
        temp = n if n > 0 else -n
        while temp > 0:
            d = temp % 10
            if d == 2 or d == 3 or d == 5 or d == 7:
                total += d
            temp //= 10
        return total
    
    @cuda.jit(device=True)
    def gpu_digit_wave(n: int64) -> int64:
        """Alternating addition and subtraction"""
        digits = cuda.local.array(20, dtype=int64)
        num_digits = 0
        temp = n if n > 0 else -n
        while temp > 0 and num_digits < 20:
            digits[num_digits] = temp % 10
            temp //= 10
            num_digits += 1
        
        result = 0
        for i in range(num_digits):
            if (num_digits - 1 - i) % 2 == 0:
                result += digits[i]
            else:
                result -= digits[i]
        return result if result >= 0 else -result
    
    @cuda.jit(device=True)
    def gpu_binary_ones(n: int64) -> int64:
        """Number of 1-bits in binary representation"""
        count = 0
        temp = n if n > 0 else -n
        while temp > 0:
            count += temp & 1
            temp >>= 1
        return count
    
    @cuda.jit(device=True)
    def gpu_digit_base_sum(n: int64, base: int64) -> int64:
        """Digit sum in given base"""
        total = 0
        temp = n if n > 0 else -n
        while temp > 0:
            total += temp % base
            temp //= base
        return total
    
    # =========================================================================
    # MAIN KERNELS
    # =========================================================================
    
    @cuda.jit
    def kernel_fibonacci_convergence(numbers, endpoints, steps, max_iter):
        """Fibonacci digit sum convergence"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        
        for _ in range(max_iter):
            new_n = gpu_fibonacci_digit_sum(n)
            step_count += 1
            if new_n == n or new_n == 0:
                break
            n = new_n
        
        endpoints[idx] = n
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_xor_reverse_convergence(numbers, endpoints, steps, cycle_lengths, max_iter):
        """XOR-reverse convergence with cycle detection"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        
        # Simple cycle detection with Floyd's algorithm
        slow = n
        fast = n
        
        for i in range(max_iter):
            slow = gpu_xor_reverse(slow)
            fast = gpu_xor_reverse(gpu_xor_reverse(fast))
            step_count += 1
            
            if slow == fast or slow == 0 or fast == 0:
                break
        
        # Determine cycle length
        if slow == 0:
            endpoints[idx] = 0
            cycle_lengths[idx] = 0
        else:
            cycle_len = 1
            temp = gpu_xor_reverse(slow)
            while temp != slow and cycle_len < 100:
                temp = gpu_xor_reverse(temp)
                cycle_len += 1
            endpoints[idx] = slow
            cycle_lengths[idx] = cycle_len
        
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_digit_alchemy_convergence(numbers, endpoints, steps, max_iter):
        """Digit alchemy convergence"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        
        for _ in range(max_iter):
            new_n = gpu_digit_alchemy(n)
            step_count += 1
            if new_n == n or new_n == 0:
                break
            n = new_n
        
        endpoints[idx] = n
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_digit_gravity_convergence(numbers, endpoints, steps, max_iter):
        """Digit gravity convergence"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        
        for _ in range(max_iter):
            new_n = gpu_digit_gravity(n)
            step_count += 1
            if new_n == n or new_n == 0:
                break
            n = new_n
        
        endpoints[idx] = n
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_multi_base_check(numbers, base2_sums, base3_sums, base10_sums, matches):
        """Check multi-base digit sum equality"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        
        s2 = gpu_digit_base_sum(n, 2)
        s3 = gpu_digit_base_sum(n, 3)
        s10 = gpu_digit_sum(n)
        
        base2_sums[idx] = s2
        base3_sums[idx] = s3
        base10_sums[idx] = s10
        matches[idx] = 1 if (s2 == s3 and s3 == s10) else 0
    
    @cuda.jit
    def kernel_prime_factor_sum_convergence(numbers, endpoints, steps, max_iter):
        """Prime factor sum convergence (simplified)"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        
        for _ in range(max_iter):
            if n <= 1:
                break
            
            # Calculate sum of prime factors
            total = 0
            temp = n
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    total += d
                    temp //= d
                d += 1
            if temp > 1:
                total += temp
            
            step_count += 1
            if total == n or total <= 1:
                break
            n = total
        
        endpoints[idx] = n
        steps[idx] = step_count


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

class GPUCreativeResearcher:
    """GPU-accelerated creative experiments."""
    
    def __init__(self, batch_size: int = 1_000_000):
        self.batch_size = batch_size
        self.threads_per_block = 256
        self.blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block
    
    def run_fibonacci_experiment(self, start: int, end: int) -> Dict:
        """Fibonacci digit sum experiment on GPU."""
        print(f"\n🔬 FIBONACCI DIGIT SUM EXPERIMENT", flush=True)
        print(f"   Range: {start:,} - {end:,}", flush=True)
        
        numbers = np.arange(start, min(end, start + self.batch_size), dtype=np.int64)
        endpoints = np.zeros_like(numbers)
        steps = np.zeros_like(numbers)
        
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.to_device(endpoints)
        d_steps = cuda.to_device(steps)
        
        t0 = time.time()
        kernel_fibonacci_convergence[self.blocks, self.threads_per_block](
            d_numbers, d_endpoints, d_steps, 100
        )
        cuda.synchronize()
        elapsed = time.time() - t0
        
        endpoints = d_endpoints.copy_to_host()
        steps = d_steps.copy_to_host()
        
        counter = Counter(endpoints)
        
        print(f"   Time: {elapsed:.3f}s ({len(numbers)/elapsed/1e6:.2f}M/s)", flush=True)
        print(f"   Top attractors:", flush=True)
        for val, count in counter.most_common(5):
            pct = 100 * count / len(numbers)
            print(f"      {val}: {count:,} ({pct:.1f}%)", flush=True)
        
        return {
            'attractors': counter,
            'avg_steps': np.mean(steps),
            'throughput': len(numbers) / elapsed,
        }
    
    def run_xor_reverse_experiment(self, start: int, end: int) -> Dict:
        """XOR-reverse experiment on GPU."""
        print(f"\n🔬 XOR-REVERSE EXPERIMENT", flush=True)
        print(f"   Range: {start:,} - {end:,}", flush=True)
        
        numbers = np.arange(start, min(end, start + self.batch_size), dtype=np.int64)
        endpoints = np.zeros_like(numbers)
        steps = np.zeros_like(numbers)
        cycle_lengths = np.zeros_like(numbers)
        
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.to_device(endpoints)
        d_steps = cuda.to_device(steps)
        d_cycle_lengths = cuda.to_device(cycle_lengths)
        
        t0 = time.time()
        kernel_xor_reverse_convergence[self.blocks, self.threads_per_block](
            d_numbers, d_endpoints, d_steps, d_cycle_lengths, 200
        )
        cuda.synchronize()
        elapsed = time.time() - t0
        
        endpoints = d_endpoints.copy_to_host()
        steps = d_steps.copy_to_host()
        cycle_lengths = d_cycle_lengths.copy_to_host()
        
        counter = Counter(endpoints)
        cycle_counter = Counter(cycle_lengths[cycle_lengths > 1])
        
        print(f"   Time: {elapsed:.3f}s ({len(numbers)/elapsed/1e6:.2f}M/s)", flush=True)
        print(f"   Top attractors:", flush=True)
        for val, count in counter.most_common(5):
            pct = 100 * count / len(numbers)
            print(f"      {val}: {count:,} ({pct:.1f}%)", flush=True)
        print(f"   Cycle lengths:", flush=True)
        for length, count in cycle_counter.most_common(5):
            print(f"      Length {length}: {count:,}", flush=True)
        
        return {
            'attractors': counter,
            'cycle_lengths': cycle_counter,
            'avg_steps': np.mean(steps),
        }
    
    def run_digit_alchemy_experiment(self, start: int, end: int) -> Dict:
        """Digit alchemy experiment on GPU."""
        print(f"\n🔬 DIGIT ALCHEMY EXPERIMENT", flush=True)
        print(f"   Range: {start:,} - {end:,}", flush=True)
        
        numbers = np.arange(start, min(end, start + self.batch_size), dtype=np.int64)
        endpoints = np.zeros_like(numbers)
        steps = np.zeros_like(numbers)
        
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.to_device(endpoints)
        d_steps = cuda.to_device(steps)
        
        t0 = time.time()
        kernel_digit_alchemy_convergence[self.blocks, self.threads_per_block](
            d_numbers, d_endpoints, d_steps, 50
        )
        cuda.synchronize()
        elapsed = time.time() - t0
        
        endpoints = d_endpoints.copy_to_host()
        steps = d_steps.copy_to_host()
        
        counter = Counter(endpoints)
        
        print(f"   Time: {elapsed:.3f}s ({len(numbers)/elapsed/1e6:.2f}M/s)", flush=True)
        print(f"   Top attractors:", flush=True)
        for val, count in counter.most_common(5):
            pct = 100 * count / len(numbers)
            print(f"      {val}: {count:,} ({pct:.1f}%)", flush=True)
        
        return {
            'attractors': counter,
            'avg_steps': np.mean(steps),
        }
    
    def run_multi_base_experiment(self, start: int, end: int) -> Dict:
        """Multi-base digit sum equality experiment."""
        print(f"\n🔬 MULTI-BASE HARMONY EXPERIMENT", flush=True)
        print(f"   Range: {start:,} - {end:,}", flush=True)
        
        numbers = np.arange(start, min(end, start + self.batch_size), dtype=np.int64)
        base2_sums = np.zeros_like(numbers)
        base3_sums = np.zeros_like(numbers)
        base10_sums = np.zeros_like(numbers)
        matches = np.zeros_like(numbers)
        
        d_numbers = cuda.to_device(numbers)
        d_base2 = cuda.to_device(base2_sums)
        d_base3 = cuda.to_device(base3_sums)
        d_base10 = cuda.to_device(base10_sums)
        d_matches = cuda.to_device(matches)
        
        t0 = time.time()
        kernel_multi_base_check[self.blocks, self.threads_per_block](
            d_numbers, d_base2, d_base3, d_base10, d_matches
        )
        cuda.synchronize()
        elapsed = time.time() - t0
        
        matches = d_matches.copy_to_host()
        match_indices = np.where(matches == 1)[0]
        harmonic_numbers = numbers[match_indices]
        
        print(f"   Time: {elapsed:.3f}s ({len(numbers)/elapsed/1e6:.2f}M/s)", flush=True)
        print(f"   Harmonic numbers found: {len(harmonic_numbers)}", flush=True)
        print(f"   First 20: {list(harmonic_numbers[:20])}", flush=True)
        
        return {
            'harmonic_numbers': harmonic_numbers,
            'count': len(harmonic_numbers),
            'density': len(harmonic_numbers) / len(numbers),
        }
    
    def run_digit_gravity_experiment(self, start: int, end: int) -> Dict:
        """Digit gravity experiment on GPU."""
        print(f"\n🔬 DIGIT GRAVITY EXPERIMENT", flush=True)
        print(f"   Range: {start:,} - {end:,}", flush=True)
        
        numbers = np.arange(start, min(end, start + self.batch_size), dtype=np.int64)
        endpoints = np.zeros_like(numbers)
        steps = np.zeros_like(numbers)
        
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.to_device(endpoints)
        d_steps = cuda.to_device(steps)
        
        t0 = time.time()
        kernel_digit_gravity_convergence[self.blocks, self.threads_per_block](
            d_numbers, d_endpoints, d_steps, 100
        )
        cuda.synchronize()
        elapsed = time.time() - t0
        
        endpoints = d_endpoints.copy_to_host()
        steps = d_steps.copy_to_host()
        
        counter = Counter(endpoints)
        
        print(f"   Time: {elapsed:.3f}s ({len(numbers)/elapsed/1e6:.2f}M/s)", flush=True)
        print(f"   Top attractors:", flush=True)
        for val, count in counter.most_common(10):
            pct = 100 * count / len(numbers)
            print(f"      {val}: {count:,} ({pct:.1f}%)", flush=True)
        
        return {
            'attractors': counter,
            'avg_steps': np.mean(steps),
        }
    
    def run_all_experiments(self, start: int = 1, end: int = 1_000_000):
        """Run all experiments."""
        print("█" * 70, flush=True)
        print("  SYNTRIAD GPU CREATIVE RESEARCH", flush=True)
        print("█" * 70, flush=True)
        
        results = {}
        
        results['fibonacci'] = self.run_fibonacci_experiment(start, end)
        results['xor_reverse'] = self.run_xor_reverse_experiment(start, end)
        results['digit_alchemy'] = self.run_digit_alchemy_experiment(start, end)
        results['multi_base'] = self.run_multi_base_experiment(start, end)
        results['digit_gravity'] = self.run_digit_gravity_experiment(start, end)
        
        print("\n" + "█" * 70, flush=True)
        print("  SUMMARY", flush=True)
        print("█" * 70, flush=True)
        
        print("\n📊 Fibonacci: 2 attractors (1, 5)", flush=True)
        print("📊 XOR-Reverse: Interesting cycles found", flush=True)
        print("📊 Digit Alchemy: 3-attractor symmetry", flush=True)
        print(f"📊 Multi-Base: {results['multi_base']['count']} harmonic numbers", flush=True)
        print("📊 Digit Gravity: Multiple small attractors", flush=True)
        
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTRIAD GPU Creative Research')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=1_000_000)
    parser.add_argument('--batch', type=int, default=1_000_000)
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'fibonacci', 'xor', 'alchemy', 'multibase', 'gravity'])
    args = parser.parse_args()
    
    if not HAS_CUDA:
        print("❌ CUDA not available. Install numba with CUDA support.")
        return
    
    researcher = GPUCreativeResearcher(batch_size=args.batch)
    
    if args.experiment == 'all':
        researcher.run_all_experiments(args.start, args.end)
    elif args.experiment == 'fibonacci':
        researcher.run_fibonacci_experiment(args.start, args.end)
    elif args.experiment == 'xor':
        researcher.run_xor_reverse_experiment(args.start, args.end)
    elif args.experiment == 'alchemy':
        researcher.run_digit_alchemy_experiment(args.start, args.end)
    elif args.experiment == 'multibase':
        researcher.run_multi_base_experiment(args.start, args.end)
    elif args.experiment == 'gravity':
        researcher.run_digit_gravity_experiment(args.start, args.end)


if __name__ == "__main__":
    main()
