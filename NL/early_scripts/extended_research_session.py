#!/usr/bin/env python3
"""
SYNTRIAD Extended Research Session v4.0
========================================

Uitgebreide onderzoekssessie (5+ minuten) die:
1. Voortbouwt op eerdere ontdekkingen
2. Creatief, iteratief en adaptief onderzoekt
3. Alles vastlegt in database en rapport
4. Automatisch dieper graaft bij interessante patronen

Ontworpen voor langdurig autonoom onderzoek.
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
from datetime import datetime

try:
    from numba import cuda, int64
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("⚠️  CUDA niet beschikbaar")

# Import GPU kernels from deep researcher
from gpu_deep_researcher import (
    kernel_pipeline_convergence, kernel_large_number_analysis,
    OP_NAMES, OP_CODES, HAS_CUDA
)


@dataclass
class SessionConfig:
    """Configuratie voor extended research session."""
    duration_minutes: float = 5.0
    batch_size: int = 2_000_000
    max_pipeline_length: int = 5
    max_iterations: int = 150
    min_convergence: float = 0.25
    initial_exploration_rate: float = 0.4
    db_path: str = "extended_research.db"
    report_path: str = "EXTENDED_RESEARCH_REPORT.md"


@dataclass 
class Discovery:
    """Een ontdekking."""
    pipeline: List[str]
    attractor: int
    convergence: float
    score: float
    digit_range: str
    timestamp: float
    verified_ranges: List[str] = field(default_factory=list)


class ExtendedResearchSession:
    """
    Uitgebreide onderzoekssessie met volledige persistentie.
    """
    
    TRIVIAL_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    KNOWN_CONSTANTS = {495, 6174, 1089, 10890, 99099, 26244}
    
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
        
        # Statistics
        self.total_numbers_tested = 0
        self.total_pipelines_tested = 0
        
        # Load previous discoveries
        self._init_database()
        self._load_previous_discoveries()
    
    def _init_database(self):
        """Initialize database."""
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
        conn.commit()
        conn.close()
    
    def _load_previous_discoveries(self):
        """Laad eerdere ontdekkingen om van te leren."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        
        # Laad operation learning state
        c.execute('SELECT op_name, alpha, beta FROM operation_learning')
        for row in c.fetchall():
            self.op_alpha[row[0]] = row[1]
            self.op_beta[row[0]] = row[2]
        
        # Laad eerdere ontdekkingen voor pair/triple learning
        c.execute('SELECT pipeline, score FROM session_discoveries ORDER BY score DESC LIMIT 100')
        for row in c.fetchall():
            pipeline_str = row[0]
            score = row[1]
            ops = pipeline_str.split(' → ')
            
            # Update pair scores
            for i in range(len(ops) - 1):
                if ops[i] in OP_CODES and ops[i+1] in OP_CODES:
                    pair = (OP_CODES[ops[i]], OP_CODES[ops[i+1]])
                    self.pair_scores[pair] = max(self.pair_scores[pair], score)
            
            # Update triple scores
            for i in range(len(ops) - 2):
                if all(ops[j] in OP_CODES for j in range(i, i+3)):
                    triple = (OP_CODES[ops[i]], OP_CODES[ops[i+1]], OP_CODES[ops[i+2]])
                    self.triple_scores[triple] = max(self.triple_scores[triple], score)
        
        conn.close()
        
        if self.op_alpha:
            print(f"📚 Geladen: {len(self.op_alpha)} operatie-statistieken", flush=True)
        if self.pair_scores:
            print(f"📚 Geladen: {len(self.pair_scores)} succesvolle paren", flush=True)
    
    def _save_discovery(self, discovery: Discovery, session_id: str):
        """Save discovery."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO session_discoveries 
            (session_id, pipeline, attractor, convergence, score, digit_range, verified_ranges, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            ' → '.join(discovery.pipeline),
            discovery.attractor,
            discovery.convergence,
            discovery.score,
            discovery.digit_range,
            json.dumps(discovery.verified_ranges),
            discovery.timestamp,
        ))
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
                          pipelines: int, numbers: int, best_score: float, exploration: float):
        """Save cycle statistics."""
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO session_stats 
            (session_id, cycle, discoveries_count, pipelines_tested, numbers_tested, 
             best_score, exploration_rate, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, cycle, discoveries, pipelines, numbers, best_score, exploration, time.time()))
        conn.commit()
        conn.close()
    
    def thompson_sample_op(self) -> int:
        """Sample operatie met Thompson sampling."""
        scores = {}
        for op_code, op_name in OP_NAMES.items():
            alpha = self.op_alpha[op_name]
            beta = self.op_beta[op_name]
            scores[op_code] = np.random.beta(alpha, beta)
        return max(scores, key=scores.get)
    
    def generate_smart_pipeline(self, length: int = None) -> List[int]:
        """Genereer pipeline met geleerde kennis."""
        if length is None:
            length = random.randint(1, self.config.max_pipeline_length)
        
        pipeline = []
        
        # Bepaal strategie
        strategy = random.random()
        
        if strategy < 0.2 and self.pair_scores:
            # Gebruik succesvolle paren
            best_pairs = sorted(self.pair_scores.items(), key=lambda x: -x[1])[:10]
            if best_pairs:
                pair = random.choice(best_pairs)[0]
                pipeline = list(pair)
                # Voeg eventueel extra operaties toe
                while len(pipeline) < length:
                    pipeline.append(self.thompson_sample_op())
        
        elif strategy < 0.4 and self.triple_scores:
            # Gebruik succesvolle triples
            best_triples = sorted(self.triple_scores.items(), key=lambda x: -x[1])[:10]
            if best_triples:
                triple = random.choice(best_triples)[0]
                pipeline = list(triple)
                while len(pipeline) < length:
                    pipeline.append(self.thompson_sample_op())
        
        else:
            # Thompson sampling
            exploration_rate = self.config.initial_exploration_rate * (0.9 ** self.cycle_count)
            
            for _ in range(length):
                if random.random() < exploration_rate:
                    op = random.randint(0, len(OP_NAMES) - 1)
                else:
                    op = self.thompson_sample_op()
                pipeline.append(op)
        
        return pipeline[:length]
    
    def evaluate_pipeline(self, pipeline: List[int], start: int, end: int) -> Optional[Discovery]:
        """Evalueer pipeline op GPU."""
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
        self.total_numbers_tested += len(numbers)
        
        # Analyse
        counter = Counter(endpoints)
        total = len(numbers)
        
        if not counter:
            return None
        
        top_val, top_count = counter.most_common(1)[0]
        convergence = top_count / total
        
        # Score
        score = self._calculate_score(counter, total, pipeline)
        
        # Is interessant?
        if convergence < self.config.min_convergence:
            return None
        if top_val in self.TRIVIAL_VALUES and convergence < 0.9:
            return None
        if score < 30:
            return None
        
        pipeline_names = [OP_NAMES[op] for op in pipeline]
        
        return Discovery(
            pipeline=pipeline_names,
            attractor=int(top_val),
            convergence=convergence,
            score=score,
            digit_range=f"{start}-{end}",
            timestamp=time.time(),
        )
    
    def _calculate_score(self, counter: Counter, total: int, pipeline: List[int]) -> float:
        """Bereken score."""
        if not counter:
            return 0.0
        
        top_val, top_count = counter.most_common(1)[0]
        convergence = top_count / total
        
        score = convergence * 50
        
        # Niet-triviaal bonus
        if top_val not in self.TRIVIAL_VALUES:
            score += 20
        
        # Grote getallen bonus
        if top_val > 1000:
            score += 10
        if top_val > 100000:
            score += 10
        
        # Palindroom bonus
        s = str(top_val)
        if s == s[::-1]:
            score += 15
        
        # Bekende constante bonus
        if top_val in self.KNOWN_CONSTANTS:
            score += 5
        
        # Nieuwe constante bonus (niet in bekende set)
        if top_val > 100 and top_val not in self.KNOWN_CONSTANTS:
            score += 10
        
        # Pipeline lengte penalty
        score -= len(pipeline) * 1.5
        
        # Meerdere attractoren bonus
        if len(counter) >= 2:
            second_count = counter.most_common(2)[1][1]
            if second_count / total > 0.1:
                score += 8
        
        return max(0, score)
    
    def verify_discovery(self, discovery: Discovery, pipeline: List[int]) -> bool:
        """Verifieer ontdekking op grotere getallen."""
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
        """Update Thompson sampling en pair/triple scores."""
        for op_name in pipeline:
            if success:
                self.op_alpha[op_name] += score / 50
            else:
                self.op_beta[op_name] += 0.3
        
        # Update pair scores
        for i in range(len(pipeline) - 1):
            if pipeline[i] in OP_CODES and pipeline[i+1] in OP_CODES:
                pair = (OP_CODES[pipeline[i]], OP_CODES[pipeline[i+1]])
                if success:
                    self.pair_scores[pair] = max(self.pair_scores[pair], score)
        
        # Update triple scores
        for i in range(len(pipeline) - 2):
            if all(pipeline[j] in OP_CODES for j in range(i, i+3)):
                triple = tuple(OP_CODES[pipeline[i+j]] for j in range(3))
                if success:
                    self.triple_scores[triple] = max(self.triple_scores[triple], score)
    
    def explore_variations(self, base_discovery: Discovery) -> List[Discovery]:
        """Exploreer variaties van succesvolle pipeline."""
        variations = []
        base_pipeline = [OP_CODES[op] for op in base_discovery.pipeline]
        
        # Voeg operatie toe aan begin/eind
        for op in range(len(OP_NAMES)):
            for new_pipeline in [[op] + base_pipeline, base_pipeline + [op]]:
                sig = str(new_pipeline)
                if sig not in self.tested_pipelines and len(new_pipeline) <= self.config.max_pipeline_length:
                    self.tested_pipelines.add(sig)
                    self.total_pipelines_tested += 1
                    
                    result = self.evaluate_pipeline(new_pipeline, 10000, 500000)
                    if result and result.score > base_discovery.score * 0.8:
                        variations.append(result)
        
        return variations
    
    def run_cycle(self, session_id: str, cycle_num: int, 
                  start: int, end: int, num_pipelines: int) -> List[Discovery]:
        """Voer één onderzoekscyclus uit."""
        cycle_discoveries = []
        best_score = 0
        
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
                
                # Verifieer belangrijke ontdekkingen
                if result.score > 50:
                    verified = self.verify_discovery(result, pipeline)
                    if verified:
                        result.score += 10  # Verificatie bonus
                
                cycle_discoveries.append(result)
                self.discoveries.append(result)
                self._save_discovery(result, session_id)
                
                best_score = max(best_score, result.score)
                
                # Print interessante ontdekkingen
                if result.score > 45:
                    print(f"\n   🎯 {' → '.join(result.pipeline)}", flush=True)
                    print(f"      Attractor: {result.attractor} ({result.convergence*100:.1f}%)", flush=True)
                    print(f"      Score: {result.score:.1f}", flush=True)
                    if result.verified_ranges:
                        print(f"      ✅ Geverifieerd: {len(result.verified_ranges)} ranges", flush=True)
                
                # Exploreer variaties van zeer goede ontdekkingen
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
            num_pipelines, end - start, best_score, exploration_rate
        )
        
        return cycle_discoveries
    
    def generate_report(self, session_id: str):
        """Genereer markdown rapport."""
        report = f"""# SYNTRIAD Extended Research Report
## Session: {session_id}

**Datum:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Duur:** {(time.time() - self.start_time) / 60:.1f} minuten  
**Getallen getest:** {self.total_numbers_tested:,}  
**Pipelines getest:** {self.total_pipelines_tested:,}  
**Ontdekkingen:** {len(self.discoveries)}

---

## 🏆 Top Ontdekkingen

"""
        # Sort by score
        top = sorted(self.discoveries, key=lambda x: -x.score)[:20]
        
        for i, d in enumerate(top, 1):
            verified = "✅" if d.verified_ranges else ""
            report += f"""### {i}. {' → '.join(d.pipeline)} {verified}
- **Attractor:** {d.attractor}
- **Convergentie:** {d.convergence*100:.1f}%
- **Score:** {d.score:.1f}
- **Geverifieerd:** {', '.join(d.verified_ranges) if d.verified_ranges else 'Nee'}

"""
        
        # Operation rankings
        report += """---

## 📊 Operatie Rankings (Thompson Sampling)

| Operatie | Succes Score |
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

## 🔗 Beste Operatie-Paren

| Paar | Score |
|------|-------|
"""
        best_pairs = sorted(self.pair_scores.items(), key=lambda x: -x[1])[:10]
        for pair, score in best_pairs:
            pair_names = f"{OP_NAMES[pair[0]]} → {OP_NAMES[pair[1]]}"
            report += f"| {pair_names} | {score:.1f} |\n"
        
        report += f"""
---

## 📈 Statistieken

- **Totaal getallen geanalyseerd:** {self.total_numbers_tested:,}
- **Throughput:** {self.total_numbers_tested / (time.time() - self.start_time) / 1e6:.1f}M/s
- **Unieke pipelines getest:** {len(self.tested_pipelines)}
- **Ontdekkingsratio:** {len(self.discoveries) / max(1, self.total_pipelines_tested) * 100:.1f}%

---

*Gegenereerd door SYNTRIAD Extended Research Session v4.0*
"""
        
        with open(self.config.report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Rapport opgeslagen: {self.config.report_path}", flush=True)
    
    def run(self):
        """Voer uitgebreide onderzoekssessie uit."""
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.start_time = time.time()
        end_time = self.start_time + self.config.duration_minutes * 60
        
        print("█" * 70, flush=True)
        print("  SYNTRIAD EXTENDED RESEARCH SESSION v4.0", flush=True)
        print("█" * 70, flush=True)
        print(f"Session ID: {session_id}", flush=True)
        print(f"Duur: {self.config.duration_minutes} minuten", flush=True)
        print(f"Batch size: {self.config.batch_size:,}", flush=True)
        print("█" * 70, flush=True)
        
        # Progressieve ranges
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
            
            # Selecteer range gebaseerd op cyclus
            range_idx = min(self.cycle_count // 2, len(ranges) - 1)
            start, end, num_pipelines = ranges[range_idx]
            
            print(f"\n{'='*60}", flush=True)
            print(f"🔬 CYCLUS {self.cycle_count} | Verstreken: {elapsed/60:.1f}m | Resterend: {remaining:.1f}m", flush=True)
            print(f"   Range: {start:,} - {end:,} | Pipelines: {num_pipelines}", flush=True)
            print(f"   Ontdekkingen tot nu: {len(self.discoveries)}", flush=True)
            print(f"{'='*60}", flush=True)
            
            cycle_discoveries = self.run_cycle(session_id, self.cycle_count, start, end, num_pipelines)
            
            print(f"\n   Cyclus compleet: {len(cycle_discoveries)} nieuwe ontdekkingen", flush=True)
            
            # Save learning state periodiek
            if self.cycle_count % 3 == 0:
                self._save_learning_state()
            
            self.cycle_count += 1
        
        # Finale
        self._save_learning_state()
        self.generate_report(session_id)
        
        print("\n" + "█" * 70, flush=True)
        print("  SESSIE COMPLEET", flush=True)
        print("█" * 70, flush=True)
        print(f"Totale duur: {(time.time() - self.start_time) / 60:.1f} minuten", flush=True)
        print(f"Totaal ontdekkingen: {len(self.discoveries)}", flush=True)
        print(f"Getallen geanalyseerd: {self.total_numbers_tested:,}", flush=True)
        print(f"Pipelines getest: {self.total_pipelines_tested}", flush=True)
        
        if self.discoveries:
            print("\n🏆 TOP 5 ONTDEKKINGEN:", flush=True)
            top = sorted(self.discoveries, key=lambda x: -x.score)[:5]
            for i, d in enumerate(top, 1):
                print(f"\n{i}. {' → '.join(d.pipeline)}", flush=True)
                print(f"   Attractor: {d.attractor} ({d.convergence*100:.1f}%) | Score: {d.score:.1f}", flush=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SYNTRIAD Extended Research Session')
    parser.add_argument('--duration', type=float, default=5.0, help='Duur in minuten')
    parser.add_argument('--batch', type=int, default=2_000_000)
    args = parser.parse_args()
    
    if not HAS_CUDA:
        print("❌ CUDA niet beschikbaar")
        return
    
    config = SessionConfig(
        duration_minutes=args.duration,
        batch_size=args.batch,
    )
    
    session = ExtendedResearchSession(config)
    session.run()


if __name__ == "__main__":
    main()
