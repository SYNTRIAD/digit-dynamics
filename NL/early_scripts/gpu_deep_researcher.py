#!/usr/bin/env python3
"""
SYNTRIAD GPU Deep Researcher v3.0
==================================

Zelf-adapterend GPU-versneld onderzoekssysteem dat:
1. Complexe operatie-combinaties onderzoekt op grote schaal
2. Zichzelf aanpast op basis van ontdekkingen
3. Dieper graaft bij interessante patronen
4. Automatisch nieuwe hypotheses genereert en test

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
import sqlite3
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict
from pathlib import Path
import random

try:
    from numba import cuda, int64, float64, boolean
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("⚠️  CUDA niet beschikbaar")


# =============================================================================
# GPU DEVICE FUNCTIONS - Uitgebreide Operatie Bibliotheek
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
    def gpu_digit_product(n: int64) -> int64:
        if n == 0:
            return 0
        product = 1
        temp = n if n > 0 else -n
        while temp > 0:
            d = temp % 10
            if d > 0:
                product *= d
            temp //= 10
        return product
    
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
    def gpu_happy_step(n: int64) -> int64:
        return gpu_digit_pow_sum(n, 2)
    
    @cuda.jit(device=True)
    def gpu_digit_pow3(n: int64) -> int64:
        return gpu_digit_pow_sum(n, 3)
    
    @cuda.jit(device=True)
    def gpu_digit_pow4(n: int64) -> int64:
        return gpu_digit_pow_sum(n, 4)
    
    @cuda.jit(device=True)
    def gpu_digit_pow5(n: int64) -> int64:
        return gpu_digit_pow_sum(n, 5)
    
    @cuda.jit(device=True)
    def gpu_xor_reverse(n: int64) -> int64:
        if n <= 0:
            return 0
        rev = gpu_reverse(n)
        return n ^ rev
    
    @cuda.jit(device=True)
    def gpu_complement_9(n: int64) -> int64:
        if n <= 0:
            return 0
        result = 0
        multiplier = 1
        temp = n
        while temp > 0:
            d = temp % 10
            result += (9 - d) * multiplier
            multiplier *= 10
            temp //= 10
        return result
    
    @cuda.jit(device=True)
    def gpu_digit_gravity(n: int64) -> int64:
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
    def gpu_fibonacci_digit_sum(n: int64) -> int64:
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
    
    @cuda.jit(device=True)
    def gpu_digit_alchemy(n: int64) -> int64:
        """Som van (cijfer × positie), positie 1-indexed van rechts."""
        if n <= 0:
            return 0
        total = 0
        position = 1
        temp = n
        while temp > 0:
            d = temp % 10
            total += d * position
            position += 1
            temp //= 10
        return total
    
    @cuda.jit(device=True)
    def gpu_prime_factor_sum(n: int64) -> int64:
        """Som van alle priemfactoren (met herhaling)."""
        if n <= 1:
            return 0
        total = 0
        temp = n
        # Factor 2
        while temp % 2 == 0:
            total += 2
            temp //= 2
        # Oneven factoren
        f = 3
        while f * f <= temp and f < 1000:  # Limiet om overflow te voorkomen
            while temp % f == 0:
                total += f
                temp //= f
            f += 2
        # Resterende priemfactor
        if temp > 1:
            total += temp
        return total
    
    @cuda.jit(device=True)
    def gpu_multi_base_harmony(n: int64) -> int64:
        """
        Check palindroom in bases 2, 3, 10. 
        Return: aantal bases waarin n palindroom is (0-3).
        Als >= 2, return n zelf (harmonisch), anders digit_sum.
        """
        if n <= 0:
            return 0
        
        harmony_count = 0
        
        # Check base 10 palindroom
        rev10 = gpu_reverse(n)
        if rev10 == n:
            harmony_count += 1
        
        # Check base 2 palindroom
        digits2 = cuda.local.array(64, dtype=int64)
        num_d2 = 0
        temp = n
        while temp > 0 and num_d2 < 64:
            digits2[num_d2] = temp % 2
            temp //= 2
            num_d2 += 1
        is_pal2 = 1
        for i in range(num_d2 // 2):
            if digits2[i] != digits2[num_d2 - 1 - i]:
                is_pal2 = 0
                break
        harmony_count += is_pal2
        
        # Check base 3 palindroom
        digits3 = cuda.local.array(40, dtype=int64)
        num_d3 = 0
        temp = n
        while temp > 0 and num_d3 < 40:
            digits3[num_d3] = temp % 3
            temp //= 3
            num_d3 += 1
        is_pal3 = 1
        for i in range(num_d3 // 2):
            if digits3[i] != digits3[num_d3 - 1 - i]:
                is_pal3 = 0
                break
        harmony_count += is_pal3
        
        # Return n als harmonisch (>=2 bases), anders digit_sum
        if harmony_count >= 2:
            return n
        return gpu_digit_sum(n)
    
    # =========================================================================
    # COMPOSITE OPERATION DISPATCHER
    # =========================================================================
    
    @cuda.jit(device=True)
    def gpu_apply_op(n: int64, op_code: int64) -> int64:
        """Pas operatie toe op basis van code."""
        if op_code == 0:
            return gpu_reverse(n)
        elif op_code == 1:
            return gpu_digit_sum(n)
        elif op_code == 2:
            return gpu_kaprekar_step(n)
        elif op_code == 3:
            return gpu_truc_1089(n)
        elif op_code == 4:
            return gpu_happy_step(n)
        elif op_code == 5:
            return gpu_digit_pow3(n)
        elif op_code == 6:
            return gpu_digit_pow4(n)
        elif op_code == 7:
            return gpu_digit_pow5(n)
        elif op_code == 8:
            return gpu_xor_reverse(n)
        elif op_code == 9:
            return gpu_complement_9(n)
        elif op_code == 10:
            return gpu_digit_gravity(n)
        elif op_code == 11:
            return gpu_fibonacci_digit_sum(n)
        elif op_code == 12:
            return gpu_swap_ends(n)
        elif op_code == 13:
            return gpu_sort_digits_desc(n)
        elif op_code == 14:
            return gpu_sort_digits_asc(n)
        elif op_code == 15:
            return gpu_digit_product(n)
        elif op_code == 16:
            return gpu_digit_alchemy(n)
        elif op_code == 17:
            return gpu_prime_factor_sum(n)
        elif op_code == 18:
            return gpu_multi_base_harmony(n)
        else:
            return n
    
    # =========================================================================
    # MAIN RESEARCH KERNELS
    # =========================================================================
    
    @cuda.jit
    def kernel_pipeline_convergence(numbers, op_sequence, num_ops, endpoints, 
                                     steps, cycle_detected, max_iter):
        """Evalueer een pipeline van operaties op grote schaal."""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        
        # Simpele cyclus detectie
        slow = n
        fast = n
        found_cycle = 0
        
        for iteration in range(max_iter):
            # Pas alle operaties in volgorde toe
            for op_idx in range(num_ops):
                slow = gpu_apply_op(slow, op_sequence[op_idx])
                if slow > 10**15 or slow < 0:
                    slow = 0
                    break
            
            # Fast pointer: 2x zo snel
            for _ in range(2):
                for op_idx in range(num_ops):
                    fast = gpu_apply_op(fast, op_sequence[op_idx])
                    if fast > 10**15 or fast < 0:
                        fast = 0
                        break
            
            step_count += 1
            
            if slow == fast or slow == 0:
                found_cycle = 1
                break
        
        endpoints[idx] = slow
        steps[idx] = step_count
        cycle_detected[idx] = found_cycle
    
    @cuda.jit
    def kernel_large_number_analysis(numbers, op_code, endpoints, steps, 
                                      overflow_count, max_iter):
        """Analyseer grote getallen (10+ cijfers) met enkele operatie."""
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        
        n = numbers[idx]
        step_count = 0
        overflowed = 0
        
        for _ in range(max_iter):
            new_n = gpu_apply_op(n, op_code)
            step_count += 1
            
            if new_n > 10**18:
                overflowed = 1
                break
            
            if new_n == n or new_n == 0:
                break
            n = new_n
        
        endpoints[idx] = n
        steps[idx] = step_count
        overflow_count[idx] = overflowed


# =============================================================================
# OPERATION REGISTRY
# =============================================================================

OP_NAMES = {
    0: 'reverse',
    1: 'digit_sum',
    2: 'kaprekar_step',
    3: 'truc_1089',
    4: 'happy_step',
    5: 'digit_pow3',
    6: 'digit_pow4',
    7: 'digit_pow5',
    8: 'xor_reverse',
    9: 'complement_9',
    10: 'digit_gravity',
    11: 'fibonacci_digit_sum',
    12: 'swap_ends',
    13: 'sort_desc',
    14: 'sort_asc',
    15: 'digit_product',
    16: 'digit_alchemy',
    17: 'prime_factor_sum',
    18: 'multi_base_harmony',
}

OP_CODES = {v: k for k, v in OP_NAMES.items()}


# =============================================================================
# SELF-ADAPTING RESEARCHER
# =============================================================================

@dataclass
class ResearchResult:
    """Resultaat van een onderzoeksrun."""
    pipeline: List[str]
    attractors: Dict[int, int]
    convergence_rate: float
    avg_steps: float
    is_interesting: bool
    score: float


@dataclass
class DeepResearchConfig:
    """Configuratie voor deep research."""
    batch_size: int = 5_000_000
    max_pipeline_length: int = 4
    max_iterations: int = 100
    min_convergence: float = 0.3
    exploration_rate: float = 0.3
    db_path: str = "gpu_deep_research.db"


class GPUDeepResearcher:
    """
    Zelf-adapterend GPU-versneld onderzoekssysteem.
    
    Kenmerken:
    - Onderzoekt complexe operatie-combinaties
    - Past zich aan op basis van succes
    - Graaft dieper bij interessante patronen
    - Schaalt naar miljarden getallen
    """
    
    TRIVIAL_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    
    def __init__(self, config: DeepResearchConfig = None):
        self.config = config or DeepResearchConfig()
        self.threads_per_block = 256
        
        # Thompson sampling voor operaties
        self.op_success = defaultdict(lambda: 1.0)
        self.op_failure = defaultdict(lambda: 1.0)
        
        # Ontdekkingen
        self.discoveries: List[ResearchResult] = []
        self.tested_pipelines: Set[str] = set()
        
        # Beste combinaties
        self.best_pairs: Dict[Tuple[int, int], float] = {}
        self.best_triples: Dict[Tuple[int, int, int], float] = {}
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS deep_discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline TEXT,
                top_attractor INTEGER,
                convergence_rate REAL,
                score REAL,
                num_tested INTEGER,
                digit_range TEXT,
                timestamp REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS operation_stats (
                op_name TEXT PRIMARY KEY,
                success_count REAL,
                failure_count REAL,
                avg_score REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def _save_discovery(self, result: ResearchResult, num_tested: int, digit_range: str):
        """Save discovery to database."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        
        top_attractor = max(result.attractors.items(), key=lambda x: x[1])[0] if result.attractors else 0
        
        c.execute('''
            INSERT INTO deep_discoveries 
            (pipeline, top_attractor, convergence_rate, score, num_tested, digit_range, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            ' → '.join(result.pipeline),
            int(top_attractor),
            result.convergence_rate,
            result.score,
            num_tested,
            digit_range,
            time.time(),
        ))
        conn.commit()
        conn.close()
    
    def thompson_sample_op(self) -> int:
        """Sample operatie met Thompson sampling."""
        scores = {}
        for op_code in OP_NAMES.keys():
            op_name = OP_NAMES[op_code]
            alpha = self.op_success[op_name]
            beta = self.op_failure[op_name]
            scores[op_code] = np.random.beta(alpha, beta)
        return max(scores, key=scores.get)
    
    def update_thompson(self, pipeline: List[str], success: bool):
        """Update Thompson sampling scores."""
        for op_name in pipeline:
            if success:
                self.op_success[op_name] += 1.0
            else:
                self.op_failure[op_name] += 0.5
    
    def generate_pipeline(self, length: int = None) -> List[int]:
        """Genereer een pipeline met Thompson sampling."""
        if length is None:
            length = random.randint(1, self.config.max_pipeline_length)
        
        pipeline = []
        for _ in range(length):
            if random.random() < self.config.exploration_rate:
                # Random exploratie
                op = random.randint(0, len(OP_NAMES) - 1)
            else:
                # Thompson sampling
                op = self.thompson_sample_op()
            pipeline.append(op)
        
        return pipeline
    
    def evaluate_pipeline(self, pipeline: List[int], start: int, end: int) -> ResearchResult:
        """Evalueer een pipeline op GPU."""
        batch_size = min(self.config.batch_size, end - start)
        blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block
        
        numbers = np.arange(start, start + batch_size, dtype=np.int64)
        endpoints = np.zeros(batch_size, dtype=np.int64)
        steps = np.zeros(batch_size, dtype=np.int64)
        cycle_detected = np.zeros(batch_size, dtype=np.int64)
        
        op_sequence = np.array(pipeline, dtype=np.int64)
        
        d_numbers = cuda.to_device(numbers)
        d_ops = cuda.to_device(op_sequence)
        d_endpoints = cuda.to_device(endpoints)
        d_steps = cuda.to_device(steps)
        d_cycles = cuda.to_device(cycle_detected)
        
        kernel_pipeline_convergence[blocks, self.threads_per_block](
            d_numbers, d_ops, len(pipeline), d_endpoints, d_steps, d_cycles,
            self.config.max_iterations
        )
        cuda.synchronize()
        
        endpoints = d_endpoints.copy_to_host()
        steps = d_steps.copy_to_host()
        
        # Analyseer resultaten
        counter = Counter(endpoints)
        total = len(numbers)
        
        # Bereken convergentie
        if counter:
            top_val, top_count = counter.most_common(1)[0]
            convergence = top_count / total
        else:
            convergence = 0
            top_val = 0
        
        # Score berekening
        score = self._calculate_score(counter, total, pipeline)
        
        # Is het interessant?
        is_interesting = (
            convergence > self.config.min_convergence and
            top_val not in self.TRIVIAL_VALUES and
            score > 30
        )
        
        pipeline_names = [OP_NAMES[op] for op in pipeline]
        
        return ResearchResult(
            pipeline=pipeline_names,
            attractors=dict(counter.most_common(10)),
            convergence_rate=convergence,
            avg_steps=np.mean(steps),
            is_interesting=is_interesting,
            score=score,
        )
    
    def _calculate_score(self, counter: Counter, total: int, pipeline: List[int]) -> float:
        """Bereken interessantheid score."""
        if not counter:
            return 0.0
        
        top_val, top_count = counter.most_common(1)[0]
        convergence = top_count / total
        
        score = 0.0
        
        # Basis score voor convergentie
        score += convergence * 50
        
        # Bonus voor niet-triviale waarden
        if top_val not in self.TRIVIAL_VALUES:
            score += 20
        
        # Bonus voor interessante getallen
        if top_val > 1000:
            score += 10
        if str(top_val) == str(top_val)[::-1]:  # Palindroom
            score += 15
        
        # Penalty voor te lange pipelines
        score -= len(pipeline) * 2
        
        # Bonus voor meerdere sterke attractoren
        if len(counter) > 1:
            second_val, second_count = counter.most_common(2)[1]
            if second_count / total > 0.1:
                score += 10
        
        return max(0, score)
    
    def deep_dive(self, pipeline: List[int], digit_ranges: List[Tuple[int, int]]) -> List[ResearchResult]:
        """Dieper onderzoek van een interessante pipeline."""
        results = []
        
        for min_digits, max_digits in digit_ranges:
            start = 10 ** (min_digits - 1)
            end = 10 ** max_digits
            
            result = self.evaluate_pipeline(pipeline, start, min(end, start + self.config.batch_size))
            results.append(result)
        
        return results
    
    def explore_combinations(self, base_pipeline: List[int], depth: int = 1) -> List[ResearchResult]:
        """Exploreer variaties van een succesvolle pipeline."""
        results = []
        
        # Voeg operatie toe aan begin
        for op in range(len(OP_NAMES)):
            new_pipeline = [op] + base_pipeline
            sig = str(new_pipeline)
            if sig not in self.tested_pipelines:
                self.tested_pipelines.add(sig)
                result = self.evaluate_pipeline(new_pipeline, 1000, 100000)
                if result.is_interesting:
                    results.append(result)
        
        # Voeg operatie toe aan eind
        for op in range(len(OP_NAMES)):
            new_pipeline = base_pipeline + [op]
            sig = str(new_pipeline)
            if sig not in self.tested_pipelines:
                self.tested_pipelines.add(sig)
                result = self.evaluate_pipeline(new_pipeline, 1000, 100000)
                if result.is_interesting:
                    results.append(result)
        
        return results
    
    def run_research_cycle(self, num_pipelines: int = 50, 
                           start: int = 10000, end: int = 10000000) -> List[ResearchResult]:
        """Voer één onderzoekscyclus uit."""
        print(f"\n{'='*60}", flush=True)
        print(f"🔬 RESEARCH CYCLE", flush=True)
        print(f"   Range: {start:,} - {end:,}", flush=True)
        print(f"   Pipelines: {num_pipelines}", flush=True)
        print(f"{'='*60}", flush=True)
        
        cycle_results = []
        interesting_count = 0
        
        t0 = time.time()
        
        for i in range(num_pipelines):
            # Genereer pipeline
            pipeline = self.generate_pipeline()
            sig = str(pipeline)
            
            if sig in self.tested_pipelines:
                continue
            self.tested_pipelines.add(sig)
            
            # Evalueer
            result = self.evaluate_pipeline(pipeline, start, end)
            cycle_results.append(result)
            
            # Update Thompson sampling
            self.update_thompson(result.pipeline, result.is_interesting)
            
            if result.is_interesting:
                interesting_count += 1
                self.discoveries.append(result)
                self._save_discovery(result, end - start, f"{start}-{end}")
                
                print(f"\n   🎯 INTERESSANT: {' → '.join(result.pipeline)}", flush=True)
                top_val = max(result.attractors.items(), key=lambda x: x[1])[0]
                print(f"      Attractor: {top_val} ({result.convergence_rate*100:.1f}%)", flush=True)
                print(f"      Score: {result.score:.1f}", flush=True)
                
                # Deep dive bij zeer interessante resultaten
                if result.score > 50:
                    print(f"      🔍 Deep diving...", flush=True)
                    deep_results = self.deep_dive(
                        [OP_CODES[op] for op in result.pipeline],
                        [(6, 6), (7, 7), (8, 8)]
                    )
                    for dr in deep_results:
                        if dr.is_interesting:
                            print(f"         Bevestigd op grotere getallen!", flush=True)
            
            if (i + 1) % 10 == 0:
                print(f"   Tested {i+1}/{num_pipelines}...", flush=True)
        
        elapsed = time.time() - t0
        
        print(f"\n   Tijd: {elapsed:.1f}s", flush=True)
        print(f"   Interessante ontdekkingen: {interesting_count}", flush=True)
        
        # Toon top operaties
        print(f"\n   📊 Top operaties (Thompson):", flush=True)
        op_scores = {}
        for op_name in OP_NAMES.values():
            alpha = self.op_success[op_name]
            beta = self.op_failure[op_name]
            op_scores[op_name] = alpha / (alpha + beta)
        
        for op, score in sorted(op_scores.items(), key=lambda x: -x[1])[:5]:
            print(f"      {op}: {score:.3f}", flush=True)
        
        return cycle_results
    
    def run_adaptive_research(self, num_cycles: int = 5, 
                              pipelines_per_cycle: int = 30) -> List[ResearchResult]:
        """Voer adaptief onderzoek uit over meerdere cycli."""
        print("█" * 70, flush=True)
        print("  SYNTRIAD GPU DEEP RESEARCHER v3.0", flush=True)
        print("█" * 70, flush=True)
        print(f"Cycli: {num_cycles}", flush=True)
        print(f"Pipelines per cyclus: {pipelines_per_cycle}", flush=True)
        print(f"Batch size: {self.config.batch_size:,}", flush=True)
        print("█" * 70, flush=True)
        
        all_results = []
        
        # Progressief grotere getallen
        ranges = [
            (1000, 100000),
            (10000, 1000000),
            (100000, 5000000),
            (1000000, 10000000),
            (10000000, 50000000),
        ]
        
        for cycle in range(num_cycles):
            start, end = ranges[min(cycle, len(ranges) - 1)]
            
            # Pas exploration rate aan
            self.config.exploration_rate = max(0.1, 0.4 - cycle * 0.05)
            
            results = self.run_research_cycle(pipelines_per_cycle, start, end)
            all_results.extend(results)
            
            # Exploreer variaties van beste resultaten
            if self.discoveries:
                best = sorted(self.discoveries, key=lambda x: -x.score)[:3]
                for disc in best:
                    pipeline_codes = [OP_CODES[op] for op in disc.pipeline]
                    variations = self.explore_combinations(pipeline_codes)
                    all_results.extend(variations)
        
        # Finale samenvatting
        print("\n" + "█" * 70, flush=True)
        print("  ONDERZOEK COMPLEET", flush=True)
        print("█" * 70, flush=True)
        print(f"Totaal ontdekkingen: {len(self.discoveries)}", flush=True)
        
        if self.discoveries:
            print("\n🏆 TOP 10 ONTDEKKINGEN:", flush=True)
            top = sorted(self.discoveries, key=lambda x: -x.score)[:10]
            for i, disc in enumerate(top, 1):
                top_val = max(disc.attractors.items(), key=lambda x: x[1])[0]
                print(f"\n{i:2d}. {' → '.join(disc.pipeline)}", flush=True)
                print(f"    Attractor: {top_val} ({disc.convergence_rate*100:.1f}%)", flush=True)
                print(f"    Score: {disc.score:.1f}", flush=True)
        
        return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTRIAD GPU Deep Researcher')
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--pipelines', type=int, default=30)
    parser.add_argument('--batch', type=int, default=2_000_000)
    args = parser.parse_args()
    
    if not HAS_CUDA:
        print("❌ CUDA niet beschikbaar")
        return
    
    config = DeepResearchConfig(batch_size=args.batch)
    researcher = GPUDeepResearcher(config)
    
    results = researcher.run_adaptive_research(
        num_cycles=args.cycles,
        pipelines_per_cycle=args.pipelines
    )
    
    print(f"\n✅ Onderzoek afgerond! {len(researcher.discoveries)} patronen ontdekt.")


if __name__ == "__main__":
    main()
