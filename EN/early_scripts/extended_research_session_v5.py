#!/usr/bin/env python3
"""
SYNTRIAD Extended Research Session v5.0
========================================

NEW FEATURES in v5.0:
1. Cycle detection - finds cycles, not just fixed points
2. Novelty bonus - rewards new, unknown attractors
3. Genetic mutation - mutates successful pipelines
4. Attractor properties - detects palindrome, prime, perfect power, etc.

Extended research session that:
- Builds on previous discoveries
- Explores creatively, iteratively and adaptively
- Logs everything to database and report
- Automatically digs deeper into interesting patterns

Designed for long-running autonomous research.
"""

import numpy as np
import time
import sqlite3
import json
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict
from pathlib import Path
import random
from datetime import datetime

try:
    from numba import cuda, int64
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("⚠️  CUDA not available")

from gpu_deep_researcher import (
    kernel_pipeline_convergence, kernel_large_number_analysis,
    OP_NAMES, OP_CODES, HAS_CUDA
)


# =============================================================================
# ATTRACTOR PROPERTY DETECTION
# =============================================================================

def is_prime(n: int) -> bool:
    """Check if n is a prime number."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_palindrome(n: int) -> bool:
    """Check if n is a palindrome."""
    s = str(n)
    return s == s[::-1]

def is_perfect_power(n: int) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """Check if n is a perfect power (a^b where b >= 2)."""
    if n < 4:
        return False, None
    for b in range(2, int(math.log2(n)) + 2):
        a = round(n ** (1/b))
        for candidate in [a-1, a, a+1]:
            if candidate > 1 and candidate ** b == n:
                return True, (candidate, b)
    return False, None

def is_repdigit(n: int) -> bool:
    """Check if n is a repdigit (111, 222, 9999, etc.)."""
    s = str(n)
    return len(set(s)) == 1

def digit_sum(n: int) -> int:
    """Calculate digit sum."""
    return sum(int(d) for d in str(n))

def get_attractor_properties(n: int) -> Dict[str, any]:
    """Analyze all properties of an attractor."""
    props = {
        'is_palindrome': is_palindrome(n),
        'is_prime': is_prime(n),
        'is_repdigit': is_repdigit(n),
        'digit_sum': digit_sum(n),
        'num_digits': len(str(n)),
    }
    
    is_power, power_info = is_perfect_power(n)
    props['is_perfect_power'] = is_power
    props['power_info'] = power_info
    
    # Check known patterns
    props['is_kaprekar_constant'] = n in {495, 6174}
    props['is_1089_family'] = n in {1089, 10890, 109890, 1098900, 10989, 99099}
    
    return props


@dataclass
class SessionConfig:
    """Configuration for extended research session."""
    duration_minutes: float = 5.0
    batch_size: int = 2_000_000
    max_pipeline_length: int = 5
    max_iterations: int = 150
    min_convergence: float = 0.25
    initial_exploration_rate: float = 0.4
    novelty_bonus: float = 15.0
    mutation_rate: float = 0.3
    db_path: str = "extended_research_v5.db"
    report_path: str = "EXTENDED_RESEARCH_REPORT_v5.md"


@dataclass 
class Discovery:
    """A discovery with extended properties."""
    pipeline: List[str]
    attractor: int
    convergence: float
    score: float
    digit_range: str
    timestamp: float
    verified_ranges: List[str] = field(default_factory=list)
    is_cycle: bool = False
    cycle_length: int = 0
    properties: Dict = field(default_factory=dict)
    is_novel: bool = False


class ExtendedResearchSessionV5:
    """
    Extended research session v5.0 with:
    - Cycle detection
    - Novelty search
    - Genetic mutation
    - Attractor property analysis
    """
    
    TRIVIAL_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    KNOWN_CONSTANTS = {495, 6174, 1089, 10890, 99099, 26244, 109890, 1098900}
    
    def __init__(self, config: SessionConfig = None):
        self.config = config or SessionConfig()
        self.threads_per_block = 256
        
        # Thompson sampling
        self.op_alpha = defaultdict(lambda: 1.0)
        self.op_beta = defaultdict(lambda: 1.0)
        
        # Pair/triple success tracking
        self.pair_scores: Dict[Tuple[int, int], float] = defaultdict(float)
        self.triple_scores: Dict[Tuple[int, int, int], float] = defaultdict(float)
        
        # Session state
        self.discoveries: List[Discovery] = []
        self.tested_pipelines: Set[str] = set()
        self.start_time = None
        self.cycle_count = 0
        
        # NEW: Novelty archive - all known attractors
        self.known_attractors: Set[int] = set()
        self.novel_attractors: Set[int] = set()
        
        # NEW: Successful pipelines for mutation
        self.elite_pipelines: List[Tuple[List[int], float]] = []
        
        # Statistics
        self.total_numbers_tested = 0
        self.total_pipelines_tested = 0
        self.cycles_found = 0
        self.novel_discoveries = 0
        
        # Load previous discoveries
        self._init_database()
        self._load_previous_discoveries()
    
    def _init_database(self):
        """Initialize database with extended schema."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS session_discoveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                pipeline TEXT,
                attractor INTEGER,
                convergence REAL,
                score REAL,
                digit_range TEXT,
                verified_ranges TEXT,
                is_cycle INTEGER DEFAULT 0,
                cycle_length INTEGER DEFAULT 0,
                properties TEXT,
                is_novel INTEGER DEFAULT 0,
                timestamp REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS session_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                cycle INTEGER,
                discoveries_count INTEGER,
                pipelines_tested INTEGER,
                numbers_tested INTEGER,
                best_score REAL,
                exploration_rate REAL,
                novel_count INTEGER DEFAULT 0,
                cycles_found INTEGER DEFAULT 0,
                timestamp REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS operation_learning (
                op_name TEXT PRIMARY KEY,
                alpha REAL,
                beta REAL,
                updated REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS known_attractors (
                attractor INTEGER PRIMARY KEY,
                first_seen REAL,
                discovery_count INTEGER DEFAULT 1
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS elite_pipelines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline TEXT,
                score REAL,
                attractor INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def _load_previous_discoveries(self):
        """Load previous discoveries to learn from."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        
        # Load operation learning state
        c.execute('SELECT op_name, alpha, beta FROM operation_learning')
        for row in c.fetchall():
            self.op_alpha[row[0]] = row[1]
            self.op_beta[row[0]] = row[2]
        
        # Load known attractors for novelty search
        c.execute('SELECT attractor FROM known_attractors')
        for row in c.fetchall():
            self.known_attractors.add(row[0])
        
        # Load elite pipelines for mutation
        c.execute('SELECT pipeline, score FROM elite_pipelines ORDER BY score DESC LIMIT 50')
        for row in c.fetchall():
            ops = row[0].split(' → ')
            if all(op in OP_CODES for op in ops):
                pipeline = [OP_CODES[op] for op in ops]
                self.elite_pipelines.append((pipeline, row[1]))
        
        # Load previous discoveries for pair/triple learning
        c.execute('SELECT pipeline, score FROM session_discoveries ORDER BY score DESC LIMIT 100')
        for row in c.fetchall():
            pipeline_str = row[0]
            score = row[1]
            ops = pipeline_str.split(' → ')
            
            for i in range(len(ops) - 1):
                if ops[i] in OP_CODES and ops[i+1] in OP_CODES:
                    pair = (OP_CODES[ops[i]], OP_CODES[ops[i+1]])
                    self.pair_scores[pair] = max(self.pair_scores[pair], score)
            
            for i in range(len(ops) - 2):
                if all(ops[j] in OP_CODES for j in range(i, i+3)):
                    triple = (OP_CODES[ops[i]], OP_CODES[ops[i+1]], OP_CODES[ops[i+2]])
                    self.triple_scores[triple] = max(self.triple_scores[triple], score)
        
        conn.close()
        
        if self.op_alpha:
            print(f"📚 Loaded: {len(self.op_alpha)} operation statistics", flush=True)
        if self.pair_scores:
            print(f"📚 Loaded: {len(self.pair_scores)} successful pairs", flush=True)
        if self.known_attractors:
            print(f"📚 Loaded: {len(self.known_attractors)} known attractors", flush=True)
        if self.elite_pipelines:
            print(f"📚 Loaded: {len(self.elite_pipelines)} elite pipelines for mutation", flush=True)
    
    def _save_discovery(self, discovery: Discovery, session_id: str):
        """Save discovery with extended properties."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO session_discoveries 
            (session_id, pipeline, attractor, convergence, score, digit_range, 
             verified_ranges, is_cycle, cycle_length, properties, is_novel, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            ' → '.join(discovery.pipeline),
            discovery.attractor,
            discovery.convergence,
            discovery.score,
            discovery.digit_range,
            json.dumps(discovery.verified_ranges),
            1 if discovery.is_cycle else 0,
            discovery.cycle_length,
            json.dumps(discovery.properties),
            1 if discovery.is_novel else 0,
            discovery.timestamp,
        ))
        
        # Update known attractors
        c.execute('''
            INSERT INTO known_attractors (attractor, first_seen, discovery_count)
            VALUES (?, ?, 1)
            ON CONFLICT(attractor) DO UPDATE SET discovery_count = discovery_count + 1
        ''', (discovery.attractor, time.time()))
        
        conn.commit()
        conn.close()
    
    def _save_elite_pipeline(self, pipeline: List[str], score: float, attractor: int):
        """Save elite pipeline for future mutation."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO elite_pipelines (pipeline, score, attractor)
            VALUES (?, ?, ?)
        ''', (' → '.join(pipeline), score, attractor))
        conn.commit()
        conn.close()
    
    def _save_learning_state(self):
        """Save Thompson sampling state."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        
        for op_name in OP_NAMES.values():
            c.execute('''
                INSERT OR REPLACE INTO operation_learning (op_name, alpha, beta, updated)
                VALUES (?, ?, ?, ?)
            ''', (op_name, self.op_alpha[op_name], self.op_beta[op_name], time.time()))
        
        conn.commit()
        conn.close()
    
    def _save_cycle_stats(self, session_id: str, cycle: int, discoveries: int, 
                          pipelines: int, numbers: int, best_score: float, 
                          exploration: float, novel_count: int, cycles_found: int):
        """Save cycle statistics."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO session_stats 
            (session_id, cycle, discoveries_count, pipelines_tested, numbers_tested, 
             best_score, exploration_rate, novel_count, cycles_found, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, cycle, discoveries, pipelines, numbers, best_score, 
              exploration, novel_count, cycles_found, time.time()))
        conn.commit()
        conn.close()
    
    def thompson_sample_op(self) -> int:
        """Sample operation using Thompson sampling."""
        scores = {}
        for op_code, op_name in OP_NAMES.items():
            alpha = self.op_alpha[op_name]
            beta = self.op_beta[op_name]
            scores[op_code] = np.random.beta(alpha, beta)
        return max(scores, key=scores.get)
    
    # =========================================================================
    # GENETIC MUTATION
    # =========================================================================
    
    def mutate_pipeline(self, pipeline: List[int]) -> List[int]:
        """Mutate a pipeline in various ways."""
        mutated = pipeline.copy()
        mutation_type = random.choice(['swap', 'replace', 'insert', 'delete', 'shuffle'])
        
        if mutation_type == 'swap' and len(mutated) >= 2:
            # Swap two operations
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        elif mutation_type == 'replace':
            # Replace één operatie
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] = random.randint(0, len(OP_NAMES) - 1)
        
        elif mutation_type == 'insert' and len(mutated) < self.config.max_pipeline_length:
            # Add operation
            idx = random.randint(0, len(mutated))
            mutated.insert(idx, self.thompson_sample_op())
        
        elif mutation_type == 'delete' and len(mutated) > 1:
            # Remove operation
            idx = random.randint(0, len(mutated) - 1)
            mutated.pop(idx)
        
        elif mutation_type == 'shuffle' and len(mutated) >= 3:
            # Shuffle a segment
            start = random.randint(0, len(mutated) - 2)
            end = random.randint(start + 2, len(mutated))
            segment = mutated[start:end]
            random.shuffle(segment)
            mutated[start:end] = segment
        
        return mutated
    
    def crossover_pipelines(self, p1: List[int], p2: List[int]) -> List[int]:
        """Combine two pipelines (crossover)."""
        if len(p1) < 2 or len(p2) < 2:
            return p1.copy()
        
        # Single-point crossover
        cut1 = random.randint(1, len(p1) - 1)
        cut2 = random.randint(1, len(p2) - 1)
        
        child = p1[:cut1] + p2[cut2:]
        return child[:self.config.max_pipeline_length]
    
    def generate_smart_pipeline(self, length: int = None) -> List[int]:
        """Generate pipeline using learned knowledge and genetic operations."""
        if length is None:
            length = random.randint(1, self.config.max_pipeline_length)
        
        pipeline = []
        strategy = random.random()
        
        # NEW: Genetic mutation of elite pipelines
        if strategy < 0.25 and self.elite_pipelines:
            # Mutate an elite pipeline
            elite, _ = random.choice(self.elite_pipelines[:20])
            pipeline = self.mutate_pipeline(elite)
        
        elif strategy < 0.35 and len(self.elite_pipelines) >= 2:
            # Crossover of two elite pipelines
            (p1, _), (p2, _) = random.sample(self.elite_pipelines[:20], 2)
            pipeline = self.crossover_pipelines(p1, p2)
        
        elif strategy < 0.5 and self.pair_scores:
            # Use successful pairs
            best_pairs = sorted(self.pair_scores.items(), key=lambda x: -x[1])[:10]
            if best_pairs:
                pair = random.choice(best_pairs)[0]
                pipeline = list(pair)
                while len(pipeline) < length:
                    pipeline.append(self.thompson_sample_op())
        
        elif strategy < 0.65 and self.triple_scores:
            # Use successful triples
            best_triples = sorted(self.triple_scores.items(), key=lambda x: -x[1])[:10]
            if best_triples:
                triple = random.choice(best_triples)[0]
                pipeline = list(triple)
                while len(pipeline) < length:
                    pipeline.append(self.thompson_sample_op())
        
        else:
            # Thompson sampling with exploration
            exploration_rate = self.config.initial_exploration_rate * (0.9 ** self.cycle_count)
            
            for _ in range(length):
                if random.random() < exploration_rate:
                    op = random.randint(0, len(OP_NAMES) - 1)
                else:
                    op = self.thompson_sample_op()
                pipeline.append(op)
        
        return pipeline[:self.config.max_pipeline_length] if pipeline else [self.thompson_sample_op()]
    
    # =========================================================================
    # CYCLE DETECTION
    # =========================================================================
    
    def detect_cycle_cpu(self, pipeline: List[int], start_val: int, max_iter: int = 100) -> Tuple[bool, int, List[int]]:
        """Detect cycle in CPU for verification."""
        seen = {}
        current = start_val
        sequence = [current]
        
        for step in range(max_iter):
            # Apply all operations
            for op_code in pipeline:
                op_name = OP_NAMES[op_code]
                current = self._apply_op_cpu(current, op_name)
                if current > 10**15:  # Overflow check
                    return False, 0, []
            
            if current in seen:
                cycle_start = seen[current]
                cycle_length = step - cycle_start
                return True, cycle_length, sequence[cycle_start:]
            
            seen[current] = step
            sequence.append(current)
        
        return False, 0, []
    
    def _apply_op_cpu(self, n: int, op_name: str) -> int:
        """Apply operation on CPU (for verification)."""
        if n <= 0:
            return 0
        
        if op_name == 'reverse':
            return int(str(n)[::-1])
        elif op_name == 'digit_sum':
            return sum(int(d) for d in str(n))
        elif op_name == 'digit_product':
            result = 1
            for d in str(n):
                if d != '0':
                    result *= int(d)
            return result
        elif op_name == 'kaprekar_step':
            digits = sorted(str(n))
            small = int(''.join(digits)) if digits[0] != '0' else int(''.join(digits).lstrip('0') or '0')
            big = int(''.join(reversed(digits)))
            return big - small
        elif op_name == 'truc_1089':
            rev = int(str(n)[::-1])
            diff = abs(n - rev)
            if diff == 0:
                return 0
            rev_diff = int(str(diff)[::-1])
            return diff + rev_diff
        elif op_name == 'happy_step':
            return sum(int(d)**2 for d in str(n))
        elif op_name.startswith('digit_pow'):
            power = int(op_name[-1])
            return sum(int(d)**power for d in str(n))
        elif op_name == 'complement_9':
            return int(''.join(str(9 - int(d)) for d in str(n)))
        elif op_name == 'digit_alchemy':
            return sum(int(d) * (i+1) for i, d in enumerate(str(n)[::-1]))
        else:
            return n
    
    def evaluate_pipeline(self, pipeline: List[int], start: int, end: int) -> Optional[Discovery]:
        """Evaluate pipeline on GPU with cycle detection."""
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
        cycles = d_cycles.copy_to_host()
        self.total_numbers_tested += len(numbers)
        
        # Analyse
        counter = Counter(endpoints)
        total = len(numbers)
        
        if not counter:
            return None
        
        top_val, top_count = counter.most_common(1)[0]
        convergence = top_count / total
        
        # NEW: Check for cycles
        cycle_ratio = np.sum(cycles) / total
        is_cycle = cycle_ratio > 0.1
        
        # NEW: Check novelty
        is_novel = top_val not in self.known_attractors and top_val not in self.TRIVIAL_VALUES
        
        # NEW: Attractor properties
        properties = get_attractor_properties(top_val) if top_val > 0 else {}
        
        # Score with all new factors
        score = self._calculate_score(counter, total, pipeline, is_novel, is_cycle, properties)
        
        # Is interesting?
        if convergence < self.config.min_convergence:
            return None
        if top_val in self.TRIVIAL_VALUES and convergence < 0.9 and not is_cycle:
            return None
        if score < 30:
            return None
        
        pipeline_names = [OP_NAMES[op] for op in pipeline]
        
        # Update novelty tracking
        if is_novel:
            self.novel_attractors.add(top_val)
            self.novel_discoveries += 1
        self.known_attractors.add(top_val)
        
        if is_cycle:
            self.cycles_found += 1
        
        return Discovery(
            pipeline=pipeline_names,
            attractor=int(top_val),
            convergence=convergence,
            score=score,
            digit_range=f"{start}-{end}",
            timestamp=time.time(),
            is_cycle=is_cycle,
            cycle_length=0,  # Would need CPU verification
            properties=properties,
            is_novel=is_novel,
        )
    
    def _calculate_score(self, counter: Counter, total: int, pipeline: List[int],
                         is_novel: bool, is_cycle: bool, properties: Dict) -> float:
        """Calculate score with all new factors."""
        if not counter:
            return 0.0
        
        top_val, top_count = counter.most_common(1)[0]
        convergence = top_count / total
        
        score = convergence * 50
        
        # Non-trivial bonus
        if top_val not in self.TRIVIAL_VALUES:
            score += 20
        
        # Large numbers bonus
        if top_val > 1000:
            score += 10
        if top_val > 100000:
            score += 10
        
        # NEW: Novelty bonus
        if is_novel:
            score += self.config.novelty_bonus
        
        # NEW: Cycle bonus
        if is_cycle:
            score += 12
        
        # NEW: Attractor property bonuses
        if properties.get('is_palindrome'):
            score += 15
        if properties.get('is_prime'):
            score += 20
        if properties.get('is_perfect_power'):
            score += 15
        if properties.get('is_repdigit'):
            score += 10
        if properties.get('is_1089_family'):
            score += 8
        
        # Known constant bonus
        if top_val in self.KNOWN_CONSTANTS:
            score += 5
        
        # Pipeline length penalty
        score -= len(pipeline) * 1.5
        
        # Multiple attractors bonus
        if len(counter) >= 2:
            second_count = counter.most_common(2)[1][1]
            if second_count / total > 0.1:
                score += 8
        
        return max(0, score)
    
    def verify_discovery(self, discovery: Discovery, pipeline: List[int]) -> bool:
        """Verify discovery on larger numbers."""
        verified = []
        
        test_ranges = [
            (100000, 1000000),
            (1000000, 5000000),
            (10000000, 20000000),
        ]
        
        for start, end in test_ranges:
            result = self.evaluate_pipeline(pipeline, start, end)
            if result and result.attractor == discovery.attractor and result.convergence > 0.3:
                verified.append(f"{start}-{end}")
        
        discovery.verified_ranges = verified
        return len(verified) >= 2
    
    def update_learning(self, pipeline: List[str], score: float, success: bool):
        """Update Thompson sampling and pair/triple scores."""
        for op_name in pipeline:
            if success:
                self.op_alpha[op_name] += score / 50
            else:
                self.op_beta[op_name] += 0.3
        
        for i in range(len(pipeline) - 1):
            if pipeline[i] in OP_CODES and pipeline[i+1] in OP_CODES:
                pair = (OP_CODES[pipeline[i]], OP_CODES[pipeline[i+1]])
                if success:
                    self.pair_scores[pair] = max(self.pair_scores[pair], score)
        
        for i in range(len(pipeline) - 2):
            if all(pipeline[j] in OP_CODES for j in range(i, i+3)):
                triple = tuple(OP_CODES[pipeline[i+j]] for j in range(3))
                if success:
                    self.triple_scores[triple] = max(self.triple_scores[triple], score)
    
    def explore_variations(self, base_discovery: Discovery) -> List[Discovery]:
        """Explore variations of a successful pipeline with mutation."""
        variations = []
        base_pipeline = [OP_CODES[op] for op in base_discovery.pipeline]
        
        # Standard variations
        for op in range(len(OP_NAMES)):
            for new_pipeline in [[op] + base_pipeline, base_pipeline + [op]]:
                sig = str(new_pipeline)
                if sig not in self.tested_pipelines and len(new_pipeline) <= self.config.max_pipeline_length:
                    self.tested_pipelines.add(sig)
                    self.total_pipelines_tested += 1
                    
                    result = self.evaluate_pipeline(new_pipeline, 10000, 500000)
                    if result and result.score > base_discovery.score * 0.8:
                        variations.append(result)
        
        # NEW: Mutation variations
        for _ in range(5):
            mutated = self.mutate_pipeline(base_pipeline)
            sig = str(mutated)
            if sig not in self.tested_pipelines:
                self.tested_pipelines.add(sig)
                self.total_pipelines_tested += 1
                
                result = self.evaluate_pipeline(mutated, 10000, 500000)
                if result and result.score > base_discovery.score * 0.7:
                    variations.append(result)
        
        return variations
    
    def run_cycle(self, session_id: str, cycle_num: int, 
                  start: int, end: int, num_pipelines: int) -> List[Discovery]:
        """Run één research cycle."""
        cycle_discoveries = []
        best_score = 0
        cycle_novel = 0
        cycle_cycles = 0
        
        for i in range(num_pipelines):
            pipeline = self.generate_smart_pipeline()
            sig = str(pipeline)
            
            if sig in self.tested_pipelines:
                continue
            
            self.tested_pipelines.add(sig)
            self.total_pipelines_tested += 1
            
            result = self.evaluate_pipeline(pipeline, start, end)
            
            if result:
                self.update_learning(result.pipeline, result.score, True)
                
                # Track novelty and cycles
                if result.is_novel:
                    cycle_novel += 1
                if result.is_cycle:
                    cycle_cycles += 1
                
                # Verify important discoveries
                if result.score > 50:
                    verified = self.verify_discovery(result, pipeline)
                    if verified:
                        result.score += 10
                
                # Add to elite pipelines
                if result.score > 70:
                    self.elite_pipelines.append((pipeline, result.score))
                    self._save_elite_pipeline(result.pipeline, result.score, result.attractor)
                    # Keep elite list limited
                    self.elite_pipelines = sorted(self.elite_pipelines, key=lambda x: -x[1])[:100]
                
                cycle_discoveries.append(result)
                self.discoveries.append(result)
                self._save_discovery(result, session_id)
                
                best_score = max(best_score, result.score)
                
                # Print interesting discoveries
                if result.score > 45:
                    novelty_tag = "🆕" if result.is_novel else ""
                    cycle_tag = "🔄" if result.is_cycle else ""
                    print(f"\n   🎯 {' → '.join(result.pipeline)} {novelty_tag}{cycle_tag}", flush=True)
                    print(f"      Attractor: {result.attractor} ({result.convergence*100:.1f}%)", flush=True)
                    print(f"      Score: {result.score:.1f}", flush=True)
                    
                    # Print properties
                    props = []
                    if result.properties.get('is_palindrome'):
                        props.append("palindrome")
                    if result.properties.get('is_prime'):
                        props.append("prime")
                    if result.properties.get('is_perfect_power'):
                        props.append(f"perfect power {result.properties.get('power_info')}")
                    if props:
                        print(f"      Properties: {', '.join(props)}", flush=True)
                    
                    if result.verified_ranges:
                        print(f"      ✅ Verified: {len(result.verified_ranges)} ranges", flush=True)
                
                # Explore variations of very good discoveries
                if result.score > 60:
                    variations = self.explore_variations(result)
                    for var in variations:
                        cycle_discoveries.append(var)
                        self.discoveries.append(var)
                        self._save_discovery(var, session_id)
            else:
                self.update_learning([OP_NAMES[op] for op in pipeline], 0, False)
        
        # Save cycle stats
        exploration_rate = self.config.initial_exploration_rate * (0.9 ** cycle_num)
        self._save_cycle_stats(
            session_id, cycle_num, len(cycle_discoveries),
            num_pipelines, end - start, best_score, exploration_rate,
            cycle_novel, cycle_cycles
        )
        
        return cycle_discoveries
    
    def generate_report(self, session_id: str):
        """Generate extended markdown report."""
        report = f"""# SYNTRIAD Extended Research Report v5.0
## Session: {session_id}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Duration:** {(time.time() - self.start_time) / 60:.1f} minutes  
**Numbers tested:** {self.total_numbers_tested:,}  
**Pipelines tested:** {self.total_pipelines_tested:,}  
**Discoveries:** {len(self.discoveries)}  
**New attractors:** {self.novel_discoveries}  
**Cycles found:** {self.cycles_found}

---

## 🆕 New Features in v5.0

1. **Cycle detection** - Finds cycles, not just fixed points
2. **Novelty search** - Rewards new, unknown attractors
3. **Genetic mutation** - Mutates and combines successful pipelines
4. **Attractor eigenschappen** - Detecteert palindrome, prime, perfect power

---

## 🏆 Top Discoveries

"""
        top = sorted(self.discoveries, key=lambda x: -x.score)[:20]
        
        for i, d in enumerate(top, 1):
            verified = "✅" if d.verified_ranges else ""
            novel = "🆕" if d.is_novel else ""
            cycle = "🔄" if d.is_cycle else ""
            
            props_list = []
            if d.properties.get('is_palindrome'):
                props_list.append("palindrome")
            if d.properties.get('is_prime'):
                props_list.append("prime")
            if d.properties.get('is_perfect_power'):
                props_list.append(f"perfect power")
            props_str = f" ({', '.join(props_list)})" if props_list else ""
            
            report += f"""### {i}. {' → '.join(d.pipeline)} {verified}{novel}{cycle}
- **Attractor:** {d.attractor}{props_str}
- **Convergence:** {d.convergence*100:.1f}%
- **Score:** {d.score:.1f}
- **Verified:** {', '.join(d.verified_ranges) if d.verified_ranges else 'No'}

"""
        
        # Novel discoveries section
        novel_discoveries = [d for d in self.discoveries if d.is_novel]
        if novel_discoveries:
            report += """---

## 🆕 New Attractors (Novelty Search)

| Attractor | Pipeline | Score |
|-----------|----------|-------|
"""
            for d in sorted(novel_discoveries, key=lambda x: -x.score)[:15]:
                report += f"| {d.attractor} | {' → '.join(d.pipeline[:3])}... | {d.score:.1f} |\n"
        
        # Cycles section
        cycle_discoveries = [d for d in self.discoveries if d.is_cycle]
        if cycle_discoveries:
            report += """
---

## 🔄 Cycles Found

| Attractor | Pipeline | Convergentie |
|-----------|----------|--------------|
"""
            for d in sorted(cycle_discoveries, key=lambda x: -x.score)[:10]:
                report += f"| {d.attractor} | {' → '.join(d.pipeline[:3])}... | {d.convergence*100:.1f}% |\n"
        
        # Operation rankings
        report += """
---

## 📊 Operation Rankings (Thompson Sampling)

| Operation | Success Score |
|----------|--------------|
"""
        op_scores = {}
        for op_name in OP_NAMES.values():
            alpha = self.op_alpha[op_name]
            beta = self.op_beta[op_name]
            op_scores[op_name] = alpha / (alpha + beta)
        
        for op, score in sorted(op_scores.items(), key=lambda x: -x[1])[:10]:
            report += f"| {op} | {score:.3f} |\n"
        
        # Best pairs
        report += """
---

## 🔗 Beste Operation-Paren

| Pair | Score |
|------|-------|
"""
        best_pairs = sorted(self.pair_scores.items(), key=lambda x: -x[1])[:10]
        for pair, score in best_pairs:
            pair_names = f"{OP_NAMES[pair[0]]} → {OP_NAMES[pair[1]]}"
            report += f"| {pair_names} | {score:.1f} |\n"
        
        report += f"""
---

## 📈 Statistics

- **Total numbers analyzed:** {self.total_numbers_tested:,}
- **Throughput:** {self.total_numbers_tested / (time.time() - self.start_time) / 1e6:.1f}M/s
- **Unique pipelines tested:** {len(self.tested_pipelines)}
- **Discovery ratio:** {len(self.discoveries) / max(1, self.total_pipelines_tested) * 100:.1f}%
- **New attractors:** {self.novel_discoveries}
- **Cycles found:** {self.cycles_found}
- **Elite pipelines:** {len(self.elite_pipelines)}

---

*Generated by SYNTRIAD Extended Research Session v5.0*
"""
        
        with open(self.config.report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Report saved: {self.config.report_path}", flush=True)
    
    def run(self):
        """Run extended research session."""
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.start_time = time.time()
        end_time = self.start_time + self.config.duration_minutes * 60
        
        print("█" * 70, flush=True)
        print("  SYNTRIAD EXTENDED RESEARCH SESSION v5.0", flush=True)
        print("  🆕 Cycle detection | Novelty Search | Genetic Mutation", flush=True)
        print("█" * 70, flush=True)
        print(f"Session ID: {session_id}", flush=True)
        print(f"Duration: {self.config.duration_minutes} minutes", flush=True)
        print(f"Batch size: {self.config.batch_size:,}", flush=True)
        print(f"Operations: {len(OP_NAMES)}", flush=True)
        print("█" * 70, flush=True)
        
        ranges = [
            (1000, 100000, 40),
            (10000, 500000, 35),
            (50000, 2000000, 30),
            (100000, 5000000, 25),
            (500000, 10000000, 20),
            (1000000, 20000000, 15),
        ]
        
        while time.time() < end_time:
            elapsed = time.time() - self.start_time
            remaining = (end_time - time.time()) / 60
            
            range_idx = min(self.cycle_count // 2, len(ranges) - 1)
            start, end, num_pipelines = ranges[range_idx]
            
            print(f"\n{'='*60}", flush=True)
            print(f"🔬 CYCLE {self.cycle_count} | Elapsed: {elapsed/60:.1f}m | Remaining: {remaining:.1f}m", flush=True)
            print(f"   Range: {start:,} - {end:,} | Pipelines: {num_pipelines}", flush=True)
            print(f"   Discoveries: {len(self.discoveries)} | New: {self.novel_discoveries} | Cycles: {self.cycles_found}", flush=True)
            print(f"{'='*60}", flush=True)
            
            cycle_discoveries = self.run_cycle(session_id, self.cycle_count, start, end, num_pipelines)
            
            print(f"\n   Cycle complete: {len(cycle_discoveries)} new discoveries", flush=True)
            
            if self.cycle_count % 3 == 0:
                self._save_learning_state()
            
            self.cycle_count += 1
        
        self._save_learning_state()
        self.generate_report(session_id)
        
        print("\n" + "█" * 70, flush=True)
        print("  SESSION COMPLETE", flush=True)
        print("█" * 70, flush=True)
        print(f"Total duration: {(time.time() - self.start_time) / 60:.1f} minutes", flush=True)
        print(f"Total discoveries: {len(self.discoveries)}", flush=True)
        print(f"New attractors: {self.novel_discoveries}", flush=True)
        print(f"Cycles found: {self.cycles_found}", flush=True)
        print(f"Numbers analyzed: {self.total_numbers_tested:,}", flush=True)
        print(f"Pipelines tested: {self.total_pipelines_tested}", flush=True)
        
        if self.discoveries:
            print("\n🏆 TOP 5 DISCOVERIES:", flush=True)
            top = sorted(self.discoveries, key=lambda x: -x.score)[:5]
            for i, d in enumerate(top, 1):
                tags = []
                if d.is_novel:
                    tags.append("🆕")
                if d.is_cycle:
                    tags.append("🔄")
                tag_str = " ".join(tags)
                print(f"\n{i}. {' → '.join(d.pipeline)} {tag_str}", flush=True)
                print(f"   Attractor: {d.attractor} ({d.convergence*100:.1f}%) | Score: {d.score:.1f}", flush=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTRIAD Extended Research Session v5.0')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration in minutes')
    parser.add_argument('--batch', type=int, default=2_000_000)
    args = parser.parse_args()
    
    if not HAS_CUDA:
        print("❌ CUDA not available")
        return
    
    config = SessionConfig(
        duration_minutes=args.duration,
        batch_size=args.batch,
    )
    
    session = ExtendedResearchSessionV5(config)
    session.run()


if __name__ == "__main__":
    main()
