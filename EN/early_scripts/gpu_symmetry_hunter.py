#!/usr/bin/env python3
"""
SYNTRIAD GPU Symmetry Hunter v1.0
=================================

GPU-accelerated symmetry discovery with CUDA kernels.
Leverages RTX 4000 Ada for massive parallel evaluation.

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
import math

try:
    from numba import cuda, int64, float64, boolean
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("⚠️  CUDA not available. Install: pip install numba")


# =============================================================================
# GPU KERNELS
# =============================================================================

if HAS_CUDA:
    
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
    def gpu_sort_digits_desc(n: int64) -> int64:
        if n <= 0:
            return n
        digits = cuda.local.array(20, dtype=int64)
        num_digits = 0
        temp = n
        while temp > 0 and num_digits < 20:
            digits[num_digits] = temp % 10
            temp //= 10
            num_digits += 1
        for i in range(num_digits):
            for j in range(i + 1, num_digits):
                if digits[j] > digits[i]:
                    digits[i], digits[j] = digits[j], digits[i]
        result = 0
        for i in range(num_digits):
            result = result * 10 + digits[i]
        return result
    
    @cuda.jit(device=True)
    def gpu_sort_digits_asc(n: int64) -> int64:
        if n <= 0:
            return n
        digits = cuda.local.array(20, dtype=int64)
        num_digits = 0
        temp = n
        while temp > 0 and num_digits < 20:
            digits[num_digits] = temp % 10
            temp //= 10
            num_digits += 1
        for i in range(num_digits):
            for j in range(i + 1, num_digits):
                if digits[j] < digits[i]:
                    digits[i], digits[j] = digits[j], digits[i]
        result = 0
        for i in range(num_digits):
            result = result * 10 + digits[i]
        return result
    
    @cuda.jit(device=True)
    def gpu_kaprekar_step(n: int64) -> int64:
        big = gpu_sort_digits_desc(n)
        small = gpu_sort_digits_asc(n)
        return big - small
    
    @cuda.jit(device=True)
    def gpu_truc_1089(n: int64) -> int64:
        if n <= 0:
            return 0
        rev = gpu_reverse(n)
        diff = n - rev if n > rev else rev - n
        if diff == 0:
            return 0
        rev_diff = gpu_reverse(diff)
        return diff + rev_diff
    
    @cuda.jit(device=True)
    def gpu_happy_step(n: int64) -> int64:
        if n <= 0:
            return 0
        total = 0
        temp = n
        while temp > 0:
            d = temp % 10
            total += d * d
            temp //= 10
        return total
    
    @cuda.jit(device=True)
    def gpu_add_reverse(n: int64) -> int64:
        if n < 0:
            n = -n
        return n + gpu_reverse(n)
    
    @cuda.jit(device=True)
    def gpu_sub_reverse(n: int64) -> int64:
        if n < 0:
            n = -n
        rev = gpu_reverse(n)
        return n - rev if n > rev else rev - n
    
    # =========================================================================
    # MAIN KERNELS
    # =========================================================================
    
    @cuda.jit
    def kernel_kaprekar_convergence(numbers, endpoints, steps, max_iter):
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        n = numbers[idx]
        step_count = 0
        for _ in range(max_iter):
            new_n = gpu_kaprekar_step(n)
            step_count += 1
            if new_n == n or new_n == 0:
                break
            n = new_n
        endpoints[idx] = n
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_1089_single(numbers, endpoints):
        """Single-shot 1089 trick: apply the trick ONCE."""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        n = numbers[idx]
        if n <= 0:
            endpoints[idx] = 0
            return
        # Step 1: |n - reverse(n)|
        rev = gpu_reverse(n)
        diff = n - rev if n > rev else rev - n
        if diff == 0:
            endpoints[idx] = 0
            return
        # Step 2: diff + reverse(diff)
        rev_diff = gpu_reverse(diff)
        endpoints[idx] = diff + rev_diff
    
    @cuda.jit
    def kernel_1089_convergence(numbers, endpoints, steps, max_iter):
        """Repeated 1089 trick until convergence (for exploration)."""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        n = numbers[idx]
        step_count = 0
        for _ in range(max_iter):
            new_n = gpu_truc_1089(n)
            step_count += 1
            if new_n == n or new_n == 0 or new_n > 10**15:
                break
            n = new_n
        endpoints[idx] = n
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_happy_convergence(numbers, endpoints, steps, is_happy, max_iter):
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        n = numbers[idx]
        slow = n
        fast = n
        step_count = 0
        for _ in range(max_iter):
            slow = gpu_happy_step(slow)
            fast = gpu_happy_step(gpu_happy_step(fast))
            step_count += 1
            if slow == 1:
                is_happy[idx] = True
                endpoints[idx] = 1
                steps[idx] = step_count
                return
            if slow == fast:
                break
        is_happy[idx] = False
        endpoints[idx] = slow
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_palindrome_convergence(numbers, endpoints, steps, converged, max_iter):
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        n = numbers[idx]
        step_count = 0
        for _ in range(max_iter):
            rev = gpu_reverse(n)
            if n == rev:
                converged[idx] = True
                endpoints[idx] = n
                steps[idx] = step_count
                return
            n = n + rev
            step_count += 1
            if n > 10**15:
                break
        converged[idx] = False
        endpoints[idx] = n
        steps[idx] = step_count
    
    @cuda.jit
    def kernel_generic_iteration(numbers, endpoints, steps, cycle_lengths, op_code, max_iter):
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        n = numbers[idx]
        slow = n
        fast = n
        step_count = 0
        
        for _ in range(max_iter):
            if op_code == 0:
                slow = gpu_kaprekar_step(slow)
            elif op_code == 1:
                slow = gpu_truc_1089(slow)
            elif op_code == 2:
                slow = gpu_happy_step(slow)
            elif op_code == 3:
                slow = gpu_digit_sum(slow)
            elif op_code == 4:
                slow = gpu_add_reverse(slow)
            elif op_code == 5:
                slow = gpu_sub_reverse(slow)
            
            for _ in range(2):
                if op_code == 0:
                    fast = gpu_kaprekar_step(fast)
                elif op_code == 1:
                    fast = gpu_truc_1089(fast)
                elif op_code == 2:
                    fast = gpu_happy_step(fast)
                elif op_code == 3:
                    fast = gpu_digit_sum(fast)
                elif op_code == 4:
                    fast = gpu_add_reverse(fast)
                elif op_code == 5:
                    fast = gpu_sub_reverse(fast)
            
            step_count += 1
            
            if slow > 10**15 or fast > 10**15:
                endpoints[idx] = slow
                steps[idx] = step_count
                cycle_lengths[idx] = -1
                return
            
            if slow == fast:
                cycle_len = 1
                temp = slow
                if op_code == 0:
                    temp = gpu_kaprekar_step(temp)
                elif op_code == 1:
                    temp = gpu_truc_1089(temp)
                elif op_code == 2:
                    temp = gpu_happy_step(temp)
                elif op_code == 3:
                    temp = gpu_digit_sum(temp)
                elif op_code == 4:
                    temp = gpu_add_reverse(temp)
                elif op_code == 5:
                    temp = gpu_sub_reverse(temp)
                
                while temp != slow and cycle_len < 100:
                    cycle_len += 1
                    if op_code == 0:
                        temp = gpu_kaprekar_step(temp)
                    elif op_code == 1:
                        temp = gpu_truc_1089(temp)
                    elif op_code == 2:
                        temp = gpu_happy_step(temp)
                    elif op_code == 3:
                        temp = gpu_digit_sum(temp)
                    elif op_code == 4:
                        temp = gpu_add_reverse(temp)
                    elif op_code == 5:
                        temp = gpu_sub_reverse(temp)
                
                endpoints[idx] = slow
                steps[idx] = step_count
                cycle_lengths[idx] = cycle_len
                return
        
        endpoints[idx] = slow
        steps[idx] = step_count
        cycle_lengths[idx] = 0


# =============================================================================
# GPU EXPERIMENT RUNNER
# =============================================================================

@dataclass
class GPUExperimentConfig:
    batch_size: int = 1_000_000
    max_iterations: int = 200
    threads_per_block: int = 256


class GPUSymmetryHunter:
    def __init__(self, config: GPUExperimentConfig = None):
        self.config = config or GPUExperimentConfig()
        
        if not HAS_CUDA:
            raise RuntimeError("CUDA is required")
        
        device = cuda.get_current_device()
        self.gpu_name = device.name.decode()
        print(f"🚀 GPU: {self.gpu_name}")
    
    def _get_blocks(self, n: int) -> int:
        return (n + self.config.threads_per_block - 1) // self.config.threads_per_block
    
    def generate_numbers(self, digit_range: Tuple[int, int], count: int, constraint: str = "none") -> np.ndarray:
        min_d, max_d = digit_range
        
        if constraint == "none":
            if min_d == max_d:
                low = 10 ** (min_d - 1)
                high = 10 ** min_d - 1
            else:
                low = 10 ** (min_d - 1)
                high = 10 ** max_d - 1
            return np.random.randint(low, high + 1, size=count, dtype=np.int64)
        
        elif constraint == "descending":
            from itertools import combinations
            numbers = []
            for combo in combinations(range(10), max_d):
                digits = sorted(combo, reverse=True)
                if digits[0] != 0:
                    numbers.append(int(''.join(map(str, digits))))
            if len(numbers) < count:
                return np.array(numbers * (count // len(numbers) + 1), dtype=np.int64)[:count]
            return np.array(np.random.choice(numbers, count, replace=False), dtype=np.int64)
        
        return self.generate_numbers(digit_range, count, "none")
    
    def run_kaprekar_experiment(self, digit_range: Tuple[int, int] = (4, 4), count: int = None) -> Dict:
        count = count or self.config.batch_size
        
        print(f"\n{'='*60}")
        print(f"🔬 KAPREKAR EXPERIMENT: {digit_range[0]}-{digit_range[1]} digits")
        print(f"   Samples: {count:,}")
        print(f"{'='*60}")
        
        numbers = self.generate_numbers(digit_range, count)
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.device_array(count, dtype=np.int64)
        d_steps = cuda.device_array(count, dtype=np.int64)
        
        blocks = self._get_blocks(count)
        
        start = time.perf_counter()
        kernel_kaprekar_convergence[blocks, self.config.threads_per_block](
            d_numbers, d_endpoints, d_steps, self.config.max_iterations
        )
        cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        endpoints = d_endpoints.copy_to_host()
        steps = d_steps.copy_to_host()
        
        endpoint_counts = Counter(endpoints)
        
        print(f"\n⏱️  GPU time: {elapsed*1000:.2f} ms")
        print(f"   Throughput: {count/elapsed/1e6:.2f} M numbers/sec")
        print(f"\n📊 Attractors:")
        
        for attractor, cnt in endpoint_counts.most_common(10):
            pct = 100 * cnt / count
            print(f"   {attractor:>10} : {cnt:>10,} ({pct:5.2f}%)")
        
        return {
            'digit_range': digit_range,
            'count': count,
            'attractors': dict(endpoint_counts.most_common(20)),
            'mean_steps': float(np.mean(steps)),
            'gpu_time': elapsed,
        }
    
    def run_1089_experiment(self, digit_range: Tuple[int, int] = (3, 3), constraint: str = "descending", count: int = None, single_shot: bool = True) -> Dict:
        count = count or min(self.config.batch_size, 100000)
        
        print(f"\n{'='*60}")
        print(f"🔬 1089 TRICK EXPERIMENT: {digit_range[0]}-{digit_range[1]} digits")
        print(f"   Constraint: {constraint}")
        print(f"   Mode: {'single-shot' if single_shot else 'iterative'}")
        print(f"{'='*60}")
        
        numbers = self.generate_numbers(digit_range, count, constraint)
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.device_array(count, dtype=np.int64)
        
        blocks = self._get_blocks(count)
        
        start = time.perf_counter()
        if single_shot:
            # Real 1089 trick: apply ONCE
            kernel_1089_single[blocks, self.config.threads_per_block](
                d_numbers, d_endpoints
            )
        else:
            # Iterative version: repeat until convergence
            d_steps = cuda.device_array(count, dtype=np.int64)
            kernel_1089_convergence[blocks, self.config.threads_per_block](
                d_numbers, d_endpoints, d_steps, self.config.max_iterations
            )
        cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        endpoints = d_endpoints.copy_to_host()
        endpoint_counts = Counter(endpoints)
        
        print(f"\n⏱️  GPU time: {elapsed*1000:.2f} ms")
        print(f"   Throughput: {count/elapsed/1e6:.2f} M numbers/sec")
        print(f"\n📊 Constants:")
        
        for const, cnt in endpoint_counts.most_common(10):
            pct = 100 * cnt / count
            print(f"   {const:>12} : {cnt:>8,} ({pct:5.2f}%)")
        
        if len(endpoint_counts) == 1:
            const = list(endpoint_counts.keys())[0]
            print(f"\n🎯 UNIVERSAL CONSTANT: {const}")
        elif endpoint_counts:
            top_const, top_cnt = endpoint_counts.most_common(1)[0]
            pct = 100 * top_cnt / count
            if pct > 95:
                print(f"\n🎯 DOMINANT CONSTANT: {top_const} ({pct:.1f}%)")
        
        return {
            'digit_range': digit_range,
            'constraint': constraint,
            'single_shot': single_shot,
            'constants': dict(endpoint_counts.most_common(20)),
            'gpu_time': elapsed,
        }
    
    def run_happy_number_experiment(self, number_range: Tuple[int, int] = (1, 10000000), count: int = None) -> Dict:
        count = count or self.config.batch_size
        
        print(f"\n{'='*60}")
        print(f"🔬 HAPPY NUMBER EXPERIMENT")
        print(f"   Range: {number_range[0]:,} - {number_range[1]:,}")
        print(f"{'='*60}")
        
        numbers = np.random.randint(number_range[0], number_range[1] + 1, size=count, dtype=np.int64)
        
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.device_array(count, dtype=np.int64)
        d_steps = cuda.device_array(count, dtype=np.int64)
        d_is_happy = cuda.device_array(count, dtype=np.bool_)
        
        blocks = self._get_blocks(count)
        
        start = time.perf_counter()
        kernel_happy_convergence[blocks, self.config.threads_per_block](
            d_numbers, d_endpoints, d_steps, d_is_happy, self.config.max_iterations
        )
        cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        is_happy = d_is_happy.copy_to_host()
        endpoints = d_endpoints.copy_to_host()
        
        happy_count = np.sum(is_happy)
        happy_pct = 100 * happy_count / count
        
        unhappy_endpoints = Counter(endpoints[~is_happy])
        
        print(f"\n⏱️  GPU time: {elapsed*1000:.2f} ms")
        print(f"\n📊 Results:")
        print(f"   Happy numbers:   {happy_count:>10,} ({happy_pct:.2f}%)")
        print(f"   Unhappy numbers: {count - happy_count:>10,} ({100-happy_pct:.2f}%)")
        print(f"\n📈 Unhappy cycles:")
        
        for cycle_val, cnt in unhappy_endpoints.most_common(5):
            print(f"   {cycle_val:>5} : {cnt:>10,}")
        
        return {
            'range': number_range,
            'happy_count': int(happy_count),
            'happy_pct': happy_pct,
            'gpu_time': elapsed,
        }
    
    def run_palindrome_experiment(self, digit_range: Tuple[int, int] = (2, 2), count: int = None, max_iter: int = 200) -> Dict:
        count = count or self.config.batch_size
        
        print(f"\n{'='*60}")
        print(f"🔬 PALINDROME / LYCHREL EXPERIMENT")
        print(f"   Digits: {digit_range[0]}-{digit_range[1]}")
        print(f"{'='*60}")
        
        numbers = self.generate_numbers(digit_range, count)
        
        d_numbers = cuda.to_device(numbers)
        d_endpoints = cuda.device_array(count, dtype=np.int64)
        d_steps = cuda.device_array(count, dtype=np.int64)
        d_converged = cuda.device_array(count, dtype=np.bool_)
        
        blocks = self._get_blocks(count)
        
        start = time.perf_counter()
        kernel_palindrome_convergence[blocks, self.config.threads_per_block](
            d_numbers, d_endpoints, d_steps, d_converged, max_iter
        )
        cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        converged = d_converged.copy_to_host()
        
        conv_count = np.sum(converged)
        conv_pct = 100 * conv_count / count
        
        lychrel_numbers = numbers[~converged]
        
        print(f"\n⏱️  GPU time: {elapsed*1000:.2f} ms")
        print(f"\n📊 Results:")
        print(f"   Converges: {conv_count:>10,} ({conv_pct:.2f}%)")
        print(f"   Lychrel:     {count - conv_count:>10,} ({100-conv_pct:.2f}%)")
        
        if len(lychrel_numbers) > 0 and len(lychrel_numbers) <= 20:
            print(f"\n🔴 Lychrel candidates: {sorted(lychrel_numbers)[:20]}")
        
        return {
            'digit_range': digit_range,
            'converged_count': int(conv_count),
            'lychrel_candidates': [int(n) for n in sorted(lychrel_numbers)[:100]],
            'gpu_time': elapsed,
        }
    
    def discover_new_constants(self, digit_ranges: List[Tuple[int, int]] = None, operations: List[int] = None, samples_per_config: int = 100000) -> List[Dict]:
        digit_ranges = digit_ranges or [(3,3), (4,4), (5,5), (6,6), (7,7)]
        operations = operations or [0, 1, 4, 5]
        op_names = ['kaprekar', '1089_truc', 'happy', 'digitsum', 'add_reverse', 'sub_reverse']
        
        print(f"\n{'='*70}")
        print(f"🔬 SYSTEMATIC CONSTANT DISCOVERY")
        print(f"{'='*70}")
        
        discoveries = []
        
        for digit_range in digit_ranges:
            for op_code in operations:
                op_name = op_names[op_code]
                numbers = self.generate_numbers(digit_range, samples_per_config)
                
                d_numbers = cuda.to_device(numbers)
                d_endpoints = cuda.device_array(samples_per_config, dtype=np.int64)
                d_steps = cuda.device_array(samples_per_config, dtype=np.int64)
                d_cycle_lens = cuda.device_array(samples_per_config, dtype=np.int64)
                
                blocks = self._get_blocks(samples_per_config)
                
                kernel_generic_iteration[blocks, self.config.threads_per_block](
                    d_numbers, d_endpoints, d_steps, d_cycle_lens,
                    np.int64(op_code), self.config.max_iterations
                )
                cuda.synchronize()
                
                endpoints = d_endpoints.copy_to_host()
                endpoint_counts = Counter(endpoints)
                
                if endpoint_counts:
                    top_val, top_cnt = endpoint_counts.most_common(1)[0]
                    pct = 100 * top_cnt / samples_per_config
                    
                    if pct > 95:
                        print(f"🎯 CONSTANT: {op_name} on {digit_range[0]}-digit → {top_val} ({pct:.1f}%)")
                        discoveries.append({
                            'operation': op_name,
                            'digit_range': digit_range,
                            'constant': int(top_val),
                            'convergence_pct': pct,
                        })
        
        print(f"\n📊 Total discovered: {len(discoveries)} universal constants")
        return discoveries


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("█" * 70)
    print("  SYNTRIAD GPU SYMMETRY HUNTER")
    print("█" * 70)
    
    if not HAS_CUDA:
        print("\n❌ CUDA not available!")
        return
    
    config = GPUExperimentConfig(batch_size=2_000_000, max_iterations=200)
    hunter = GPUSymmetryHunter(config)
    
    # Kaprekar
    for digits in [3, 4, 5, 6]:
        hunter.run_kaprekar_experiment((digits, digits), count=500_000)
    
    # 1089 - with strictly descending digits (classic trick)
    print("\n" + "="*70)
    print("  1089 TRICK: STRICTLY DESCENDING DIGITS")
    print("="*70)
    for digits in [3, 4, 5, 6, 7, 8]:
        hunter.run_1089_experiment((digits, digits), constraint="descending", count=50_000, single_shot=True)
    
    # 1089 - with random numbers (does it also work?)
    print("\n" + "="*70)
    print("  1089 TRICK: RANDOM NUMBERS")
    print("="*70)
    for digits in [3, 4, 5]:
        hunter.run_1089_experiment((digits, digits), constraint="none", count=500_000, single_shot=True)
    
    # Happy numbers
    hunter.run_happy_number_experiment((1, 10_000_000), count=2_000_000)
    
    # Palindrome
    hunter.run_palindrome_experiment((2, 2), count=1_000_000, max_iter=500)
    
    # Discover
    hunter.discover_new_constants()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
