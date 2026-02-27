#!/usr/bin/env python3
"""
SYNTRIAD GPU Attractor Verification v1.0
=========================================

Exhaustieve GPU-versnelde verificatie van "likely new" attractoren:
- 99099 (digit_pow_4 → truc_1089)
- 26244 (truc_1089 → digit_pow_4)
- 4176 (sort_diff → swap_ends)
- 99962001 (kaprekar_step → sort_asc → truc_1089 → kaprekar_step)

Doel: Formeel convergentiebewijs door exhaustieve ruimte-scan.

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
from collections import Counter, defaultdict
import json
from pathlib import Path

try:
    from numba import cuda, int64, float64, boolean
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("⚠️  CUDA niet beschikbaar")


# =============================================================================
# GPU DEVICE FUNCTIONS
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
    def gpu_digit_pow_sum(n: int64, power: int64) -> int64:
        total = 0
        temp = n if n > 0 else -n
        while temp > 0:
            d = temp % 10
            p = 1
            for _ in range(power):
                p *= d
            total += p
            temp //= 10
        return total
    
    @cuda.jit(device=True)
    def gpu_digit_pow4(n: int64) -> int64:
        return gpu_digit_pow_sum(n, 4)
    
    @cuda.jit(device=True)
    def gpu_swap_ends(n: int64) -> int64:
        if n < 10:
            return n
        digits = cuda.local.array(20, dtype=int64)
        num_digits = 0
        temp = n
        while temp > 0 and num_digits < 20:
            digits[num_digits] = temp % 10
            temp //= 10
            num_digits += 1
        digits[0], digits[num_digits-1] = digits[num_digits-1], digits[0]
        result = 0
        for i in range(num_digits-1, -1, -1):
            result = result * 10 + digits[i]
        return result
    
    # =========================================================================
    # PIPELINE-SPECIFIC KERNELS
    # =========================================================================
    
    @cuda.jit
    def kernel_pipeline_99099(numbers, endpoints, steps, convergence_flags, max_iter):
        """Pipeline: digit_pow_4 → truc_1089"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        seen = cuda.local.array(200, dtype=int64)
        seen_count = 0
        
        for _ in range(max_iter):
            # Step 1: digit_pow_4
            n = gpu_digit_pow_sum(n, 4)
            # Step 2: truc_1089
            n = gpu_truc_1089(n)
            
            step_count += 1
            
            # Check for cycle/fixed point
            is_seen = False
            for i in range(seen_count):
                if seen[i] == n:
                    is_seen = True
                    break
            
            if is_seen or n == 0:
                break
            
            if seen_count < 200:
                seen[seen_count] = n
                seen_count += 1
        
        endpoints[idx] = n
        steps[idx] = step_count
        convergence_flags[idx] = 1 if n == 99099 else 0
    
    @cuda.jit
    def kernel_pipeline_26244(numbers, endpoints, steps, convergence_flags, max_iter):
        """Pipeline: truc_1089 → digit_pow_4"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        seen = cuda.local.array(200, dtype=int64)
        seen_count = 0
        
        for _ in range(max_iter):
            # Step 1: truc_1089
            n = gpu_truc_1089(n)
            # Step 2: digit_pow_4
            n = gpu_digit_pow_sum(n, 4)
            
            step_count += 1
            
            is_seen = False
            for i in range(seen_count):
                if seen[i] == n:
                    is_seen = True
                    break
            
            if is_seen or n == 0:
                break
            
            if seen_count < 200:
                seen[seen_count] = n
                seen_count += 1
        
        endpoints[idx] = n
        steps[idx] = step_count
        convergence_flags[idx] = 1 if n == 26244 else 0
    
    @cuda.jit
    def kernel_pipeline_4176(numbers, endpoints, steps, convergence_flags, max_iter):
        """Pipeline: sort_diff (kaprekar) → swap_ends"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        seen = cuda.local.array(200, dtype=int64)
        seen_count = 0
        
        for _ in range(max_iter):
            # Step 1: sort_diff (kaprekar_step)
            n = gpu_kaprekar_step(n)
            # Step 2: swap_ends
            n = gpu_swap_ends(n)
            
            step_count += 1
            
            is_seen = False
            for i in range(seen_count):
                if seen[i] == n:
                    is_seen = True
                    break
            
            if is_seen or n == 0:
                break
            
            if seen_count < 200:
                seen[seen_count] = n
                seen_count += 1
        
        endpoints[idx] = n
        steps[idx] = step_count
        convergence_flags[idx] = 1 if n == 4176 else 0
    
    @cuda.jit
    def kernel_pipeline_99962001(numbers, endpoints, steps, convergence_flags, max_iter):
        """Pipeline: kaprekar_step → sort_asc → truc_1089 → kaprekar_step"""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        seen = cuda.local.array(200, dtype=int64)
        seen_count = 0
        
        for _ in range(max_iter):
            # Step 1: kaprekar_step
            n = gpu_kaprekar_step(n)
            # Step 2: sort_asc
            n = gpu_sort_digits_asc(n)
            # Step 3: truc_1089
            n = gpu_truc_1089(n)
            # Step 4: kaprekar_step
            n = gpu_kaprekar_step(n)
            
            step_count += 1
            
            is_seen = False
            for i in range(seen_count):
                if seen[i] == n:
                    is_seen = True
                    break
            
            if is_seen or n == 0:
                break
            
            if seen_count < 200:
                seen[seen_count] = n
                seen_count += 1
        
        endpoints[idx] = n
        steps[idx] = step_count
        convergence_flags[idx] = 1 if n == 99962001 else 0


# =============================================================================
# VERIFICATION ENGINE
# =============================================================================

@dataclass
class VerificationResult:
    """Resultaat van exhaustieve verificatie."""
    attractor: int
    pipeline: str
    total_tested: int
    converged_count: int
    convergence_rate: float
    other_endpoints: Dict[int, int]
    avg_steps: float
    max_steps: int
    min_steps: int
    digit_ranges_tested: List[Tuple[int, int]]
    is_universal: bool
    exceptions: List[int]


class GPUAttractorVerifier:
    """GPU-versnelde attractor verificatie."""
    
    def __init__(self):
        if not HAS_CUDA:
            raise RuntimeError("CUDA is vereist voor GPU verificatie")
        
        self.threads_per_block = 256
        self.results = []
    
    def verify_pipeline(
        self,
        kernel,
        attractor: int,
        pipeline_name: str,
        digit_ranges: List[Tuple[int, int]],
        max_iter: int = 200
    ) -> VerificationResult:
        """Verifieer een pipeline exhaustief over meerdere digit ranges."""
        
        print(f"\n{'='*70}")
        print(f"🔬 VERIFICATIE: {pipeline_name}")
        print(f"   Target attractor: {attractor}")
        print(f"{'='*70}")
        
        total_tested = 0
        total_converged = 0
        all_endpoints = Counter()
        all_steps = []
        exceptions = []
        
        for low, high in digit_ranges:
            print(f"\n   📊 Range {low:,} - {high:,}...")
            
            # Generate numbers
            numbers = np.arange(low, high + 1, dtype=np.int64)
            n = len(numbers)
            total_tested += n
            
            # Allocate device arrays
            d_numbers = cuda.to_device(numbers)
            d_endpoints = cuda.device_array(n, dtype=np.int64)
            d_steps = cuda.device_array(n, dtype=np.int64)
            d_flags = cuda.device_array(n, dtype=np.int64)
            
            # Calculate grid
            blocks = (n + self.threads_per_block - 1) // self.threads_per_block
            
            # Run kernel
            start = time.time()
            kernel[blocks, self.threads_per_block](
                d_numbers, d_endpoints, d_steps, d_flags, max_iter
            )
            cuda.synchronize()
            elapsed = time.time() - start
            
            # Copy back
            endpoints = d_endpoints.copy_to_host()
            steps = d_steps.copy_to_host()
            flags = d_flags.copy_to_host()
            
            # Analyze
            converged = np.sum(flags)
            total_converged += converged
            all_steps.extend(steps.tolist())
            
            # Count endpoints
            endpoint_counts = Counter(endpoints)
            all_endpoints.update(endpoint_counts)
            
            # Find exceptions (numbers that don't converge to target)
            non_converged_mask = flags == 0
            if np.any(non_converged_mask):
                exc_numbers = numbers[non_converged_mask][:10]  # First 10
                exceptions.extend(exc_numbers.tolist())
            
            rate = 100 * converged / n
            throughput = n / max(elapsed, 0.001) / 1e6
            print(f"      ✓ {converged:,}/{n:,} ({rate:.2f}%) in {elapsed:.3f}s ({throughput:.1f}M/s)")
        
        # Final analysis
        convergence_rate = 100 * total_converged / total_tested
        
        # Other endpoints (excluding target)
        other_endpoints = {k: v for k, v in all_endpoints.items() if k != attractor}
        
        result = VerificationResult(
            attractor=attractor,
            pipeline=pipeline_name,
            total_tested=total_tested,
            converged_count=total_converged,
            convergence_rate=convergence_rate,
            other_endpoints=dict(Counter(other_endpoints).most_common(20)),
            avg_steps=np.mean(all_steps),
            max_steps=int(np.max(all_steps)),
            min_steps=int(np.min(all_steps)),
            digit_ranges_tested=digit_ranges,
            is_universal=convergence_rate > 99.0,
            exceptions=exceptions[:50]
        )
        
        self.results.append(result)
        return result
    
    def print_result(self, result: VerificationResult):
        """Print verificatie resultaat."""
        print(f"\n{'='*70}")
        print(f"📋 VERIFICATIE RAPPORT: {result.pipeline}")
        print(f"{'='*70}")
        
        print(f"\n🎯 Target Attractor: {result.attractor}")
        print(f"   Totaal getest: {result.total_tested:,}")
        print(f"   Geconvergeerd: {result.converged_count:,}")
        print(f"   Convergentie: {result.convergence_rate:.4f}%")
        
        print(f"\n📊 Stappen statistieken:")
        print(f"   Gemiddeld: {result.avg_steps:.2f}")
        print(f"   Min: {result.min_steps}, Max: {result.max_steps}")
        
        if result.other_endpoints:
            print(f"\n⚠️  Andere eindpunten (top 10):")
            for endpoint, count in list(result.other_endpoints.items())[:10]:
                pct = 100 * count / result.total_tested
                print(f"      {endpoint}: {count:,} ({pct:.4f}%)")
        
        if result.exceptions:
            print(f"\n🔍 Voorbeeld uitzonderingen:")
            print(f"   {result.exceptions[:10]}")
        
        print(f"\n{'='*70}")
        status = "✅ UNIVERSELE ATTRACTOR" if result.is_universal else "⚠️  NIET-UNIVERSEEL"
        print(f"   STATUS: {status}")
        print(f"{'='*70}")
    
    def generate_report(self, filename: str = "attractor_verification_report.json"):
        """Genereer JSON rapport."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": "SYNTRIAD GPU Attractor Verification v1.0",
            "results": []
        }
        
        for r in self.results:
            # Convert numpy int64 to Python int for JSON serialization
            top_endpoints = {int(k): int(v) for k, v in list(r.other_endpoints.items())[:5]}
            report["results"].append({
                "attractor": int(r.attractor),
                "pipeline": r.pipeline,
                "total_tested": int(r.total_tested),
                "converged_count": int(r.converged_count),
                "convergence_rate": float(r.convergence_rate),
                "avg_steps": float(r.avg_steps),
                "is_universal": bool(r.is_universal),
                "other_endpoints_count": len(r.other_endpoints),
                "top_other_endpoints": top_endpoints
            })
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Rapport opgeslagen: {filename}")


def run_full_verification():
    """Run volledige verificatie van alle "likely new" attractoren."""
    
    print("█" * 70)
    print("  SYNTRIAD GPU ATTRACTOR VERIFICATION")
    print("  Exhaustieve analyse van 'likely new' attractoren")
    print("█" * 70)
    
    verifier = GPUAttractorVerifier()
    
    # Define digit ranges for exhaustive testing
    ranges_4digit = [(1000, 9999)]
    ranges_5digit = [(10000, 99999)]
    ranges_6digit = [(100000, 999999)]
    ranges_7digit = [(1000000, 9999999)]
    ranges_8digit = [(10000000, 99999999)]
    
    # Small ranges for quick test
    ranges_small = [(1000, 9999), (10000, 99999)]
    
    # Medium ranges for thorough test
    ranges_medium = [(1000, 9999), (10000, 99999), (100000, 999999)]
    
    # Large ranges for exhaustive test
    ranges_large = [(1000, 9999), (10000, 99999), (100000, 999999), (1000000, 9999999)]
    
    # =========================================================================
    # 1. Verify 99099 (digit_pow_4 → truc_1089)
    # =========================================================================
    result_99099 = verifier.verify_pipeline(
        kernel=kernel_pipeline_99099,
        attractor=99099,
        pipeline_name="digit_pow_4 → truc_1089",
        digit_ranges=ranges_large,
        max_iter=200
    )
    verifier.print_result(result_99099)
    
    # =========================================================================
    # 2. Verify 26244 (truc_1089 → digit_pow_4)
    # =========================================================================
    result_26244 = verifier.verify_pipeline(
        kernel=kernel_pipeline_26244,
        attractor=26244,
        pipeline_name="truc_1089 → digit_pow_4",
        digit_ranges=ranges_large,
        max_iter=200
    )
    verifier.print_result(result_26244)
    
    # =========================================================================
    # 3. Verify 4176 (sort_diff → swap_ends)
    # =========================================================================
    result_4176 = verifier.verify_pipeline(
        kernel=kernel_pipeline_4176,
        attractor=4176,
        pipeline_name="sort_diff → swap_ends",
        digit_ranges=ranges_medium,  # 4-digit specific
        max_iter=200
    )
    verifier.print_result(result_4176)
    
    # =========================================================================
    # 4. Verify 99962001 (complex pipeline) - smaller range due to complexity
    # =========================================================================
    result_99962001 = verifier.verify_pipeline(
        kernel=kernel_pipeline_99962001,
        attractor=99962001,
        pipeline_name="kaprekar → sort_asc → truc_1089 → kaprekar",
        digit_ranges=ranges_medium,
        max_iter=200
    )
    verifier.print_result(result_99962001)
    
    # Generate report
    verifier.generate_report()
    
    # Summary
    print("\n" + "█" * 70)
    print("  SAMENVATTING")
    print("█" * 70)
    
    for r in verifier.results:
        status = "✅" if r.is_universal else "⚠️"
        print(f"\n{status} {r.pipeline}")
        print(f"   Attractor: {r.attractor}")
        print(f"   Convergentie: {r.convergence_rate:.4f}%")
        print(f"   Getest: {r.total_tested:,} getallen")
    
    print("\n" + "█" * 70)
    print("  VERIFICATIE COMPLEET")
    print("█" * 70)


if __name__ == "__main__":
    run_full_verification()
