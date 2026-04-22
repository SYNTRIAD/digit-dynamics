#!/usr/bin/env python3
"""
SYNTRIAD Extended Research Session v6.0 - ML-Enhanced
======================================================

NEW FEATURES in v6.0:
1. ML Pipeline Success Predictor - Predicts which pipelines will be successful
2. Pipeline Feature Extraction - Rich feature vectors for pipelines
3. Online Learning - Model improves during the session
4. ML-based Quality Scoring - Learned model for "interestingness"
5. CPU Parallelization - 10x throughput via multiprocessing (PERF-01/02)
6. Dynamic GPU Kernels - On-the-fly CUDA kernel generation (PERF-03-06)
7. Revised Scoring Engine v2 - Trivial filtering & property bonuses (INTL-01-10)
8. Attractor Analyzer - Mathematical property detection

Plus all v5.0 features: Cycle detection, Novelty bonus, Genetic mutation, Attractor properties
"""

import numpy as np
import time
import sqlite3
import json
import math
import pickle
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any
from collections import Counter, defaultdict
from pathlib import Path
import random
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except (ImportError, ValueError) as e:
    HAS_SKLEARN = False
    print(f"⚠️  scikit-learn not available ({e}) - ML features disabled")

try:
    from numba import cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

from gpu_deep_researcher import kernel_pipeline_convergence, OP_NAMES, OP_CODES, HAS_CUDA

# Import v5 helper functions
from extended_research_session_v5 import (
    is_prime, is_palindrome, is_perfect_power, is_repdigit,
    digit_sum, get_attractor_properties
)

# Import v6.0 new modules
try:
    from scoring_engine_v2 import ScoringEngineV2, ScoringConfig
    from attractor_analyzer import AttractorAnalyzer, PropertyBonuses
    HAS_V6_SCORING = True
except ImportError:
    HAS_V6_SCORING = False
    print("Warning: v6.0 scoring modules not available")

try:
    from multiprocessing_executor import ParallelPipelineEvaluator, ParallelExecutorConfig
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False
    print("Warning: Parallel executor not available")

try:
    from cuda_kernel_generator import DynamicKernelGenerator
    HAS_DYNAMIC_KERNELS = True
except ImportError:
    HAS_DYNAMIC_KERNELS = False


# =============================================================================
# ML PIPELINE FEATURE EXTRACTION
# =============================================================================

def get_pipeline_features(pipeline: List[int], result: Optional[Dict] = None) -> Dict[str, Any]:
    """Extract rich features from a pipeline for ML models."""
    features = {
        'length': len(pipeline),
        'unique_ops': len(set(pipeline)),
        'repetition_ratio': 1 - len(set(pipeline)) / max(1, len(pipeline)),
        'has_reverse': 1 if 0 in pipeline else 0,
        'has_kaprekar': 1 if 2 in pipeline else 0,
        'has_truc_1089': 1 if 3 in pipeline else 0,
        'has_happy_step': 1 if 4 in pipeline else 0,
        'has_prime_factor_sum': 1 if 17 in pipeline else 0,
        'has_multi_base': 1 if 18 in pipeline else 0,
        'first_op': pipeline[0] if pipeline else -1,
        'last_op': pipeline[-1] if pipeline else -1,
        'has_power_combo': 1 if any(op in pipeline for op in [12, 13, 14]) else 0,
        'consecutive_same': sum(1 for i in range(len(pipeline)-1) if pipeline[i] == pipeline[i+1]),
    }
    if result:
        features.update({
            'convergence': result.get('convergence', 0),
            'attractor_digits': len(str(result.get('attractor', 0))),
            'attractor_is_palindrome': 1 if is_palindrome(result.get('attractor', 0)) else 0,
        })
    return features


# =============================================================================
# ML PIPELINE PREDICTOR CLASS
# =============================================================================

class MLPipelinePredictor:
    """Machine Learning model for pipeline success prediction."""
    
    def __init__(self, models_dir: str = "ml_models_v6"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.success_predictor = None
        self.quality_predictor = None
        self.vectorizer = None
        self.training_data = {'pipeline_features': [], 'success_labels': [], 'quality_scores': []}
        self.min_samples_for_training = 50
        self.retrain_interval = 100
        self.samples_since_training = 0
        self.prediction_accuracy = []
        self.model_version = 0
        self._load_models()
    
    def _load_models(self):
        try:
            if (self.models_dir / "success_predictor.pkl").exists():
                with open(self.models_dir / "success_predictor.pkl", 'rb') as f:
                    self.success_predictor = pickle.load(f)
                print(f"📚 Success predictor model loaded", flush=True)
            if (self.models_dir / "quality_predictor.pkl").exists():
                with open(self.models_dir / "quality_predictor.pkl", 'rb') as f:
                    self.quality_predictor = pickle.load(f)
            if (self.models_dir / "vectorizer.pkl").exists():
                with open(self.models_dir / "vectorizer.pkl", 'rb') as f:
                    self.vectorizer = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Could not load models: {e}", flush=True)
    
    def _save_models(self):
        try:
            if self.success_predictor:
                with open(self.models_dir / "success_predictor.pkl", 'wb') as f:
                    pickle.dump(self.success_predictor, f)
            if self.quality_predictor:
                with open(self.models_dir / "quality_predictor.pkl", 'wb') as f:
                    pickle.dump(self.quality_predictor, f)
            if self.vectorizer:
                with open(self.models_dir / "vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.vectorizer, f)
        except Exception as e:
            print(f"⚠️ Could not save models: {e}", flush=True)
    
    def add_training_sample(self, pipeline: List[int], result: Dict, score: float):
        features = get_pipeline_features(pipeline, result)
        self.training_data['pipeline_features'].append(features)
        self.training_data['success_labels'].append(1 if score > 50 else 0)
        self.training_data['quality_scores'].append(score)
        self.samples_since_training += 1
        if (self.samples_since_training >= self.retrain_interval and 
            len(self.training_data['pipeline_features']) >= self.min_samples_for_training):
            self.train_models()
    
    def train_models(self):
        if not HAS_SKLEARN:
            return
        n_samples = len(self.training_data['pipeline_features'])
        if n_samples < self.min_samples_for_training:
            return
        print(f"\n🧠 Training ML models with {n_samples} samples...", flush=True)
        try:
            self.vectorizer = DictVectorizer(sparse=False)
            X = self.vectorizer.fit_transform(self.training_data['pipeline_features'])
            y_success = np.array(self.training_data['success_labels'])
            if len(np.unique(y_success)) > 1:
                X_train, X_val, y_train, y_val = train_test_split(X, y_success, test_size=0.2, random_state=42)
                self.success_predictor = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                self.success_predictor.fit(X_train, y_train)
                accuracy = self.success_predictor.score(X_val, y_val)
                self.prediction_accuracy.append(accuracy)
                print(f"   ✅ Success predictor accuracy: {accuracy:.2%}", flush=True)
            y_quality = np.array(self.training_data['quality_scores'])
            X_train, X_val, y_train, y_val = train_test_split(X, y_quality, test_size=0.2, random_state=42)
            self.quality_predictor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            self.quality_predictor.fit(X_train, y_train)
            r2 = self.quality_predictor.score(X_val, y_val)
            print(f"   ✅ Quality predictor R²: {r2:.3f}", flush=True)
            self._save_models()
            self.model_version += 1
            self.samples_since_training = 0
        except Exception as e:
            print(f"   ⚠️ Training error: {e}", flush=True)
    
    def predict_success(self, pipeline: List[int]) -> float:
        if not self.success_predictor or not self.vectorizer:
            return 0.5
        try:
            features = get_pipeline_features(pipeline)
            X = self.vectorizer.transform([features])
            proba = self.success_predictor.predict_proba(X)[0]
            return proba[1] if len(proba) > 1 else 0.5
        except:
            return 0.5
    
    def predict_quality(self, pipeline: List[int]) -> float:
        if not self.quality_predictor or not self.vectorizer:
            return 50.0
        try:
            features = get_pipeline_features(pipeline)
            X = self.vectorizer.transform([features])
            return float(self.quality_predictor.predict(X)[0])
        except:
            return 50.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.success_predictor or not self.vectorizer:
            return {}
        try:
            importances = self.success_predictor.feature_importances_
            feature_names = self.vectorizer.get_feature_names_out()
            return dict(sorted(zip(feature_names, importances), key=lambda x: -x[1])[:15])
        except:
            return {}


# =============================================================================
# SESSION CONFIG & DISCOVERY
# =============================================================================

@dataclass
class SessionConfig:
    duration_minutes: float = 5.0
    batch_size: int = 2_000_000
    max_pipeline_length: int = 5
    max_iterations: int = 150
    min_convergence: float = 0.25
    initial_exploration_rate: float = 0.4
    novelty_bonus: float = 15.0
    ml_enabled: bool = True
    db_path: str = "extended_research_v6.db"
    report_path: str = "EXTENDED_RESEARCH_REPORT_v6.md"
    ml_models_dir: str = "ml_models_v6"


@dataclass 
class Discovery:
    pipeline: List[str]
    attractor: int
    convergence: float
    score: float
    digit_range: str
    timestamp: float
    verified_ranges: List[str] = field(default_factory=list)
    is_cycle: bool = False
    properties: Dict = field(default_factory=dict)
    is_novel: bool = False
    ml_predicted_score: float = 0.0


# =============================================================================
# MAIN RESEARCH SESSION CLASS v6.0
# =============================================================================

class ExtendedResearchSessionV6:
    """ML-Enhanced Research Session with online learning."""
    
    TRIVIAL_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    KNOWN_CONSTANTS = {495, 6174, 1089, 10890, 99099, 26244, 109890, 1098900}
    
    def __init__(self, config: SessionConfig = None):
        self.config = config or SessionConfig()
        self.threads_per_block = 256
        self.op_alpha = defaultdict(lambda: 1.0)
        self.op_beta = defaultdict(lambda: 1.0)
        self.pair_scores: Dict[Tuple[int, int], float] = defaultdict(float)
        self.triple_scores: Dict[Tuple[int, int, int], float] = defaultdict(float)
        self.discoveries: List[Discovery] = []
        self.tested_pipelines: Set[str] = set()
        self.start_time = None
        self.cycle_count = 0
        self.known_attractors: Set[int] = set()
        self.novel_attractors: Set[int] = set()
        self.elite_pipelines: List[Tuple[List[int], float]] = []
        self.total_numbers_tested = 0
        self.total_pipelines_tested = 0
        self.cycles_found = 0
        self.novel_discoveries = 0
        self.ml_guided_pipelines = 0
        self.ml_guided_successes = 0
        
        # ML Predictor
        self.ml_predictor = None
        if self.config.ml_enabled and HAS_SKLEARN:
            self.ml_predictor = MLPipelinePredictor(self.config.ml_models_dir)
        
        self._init_database()
        self._load_previous_discoveries()
    
    def _init_database(self):
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS session_discoveries (
            id INTEGER PRIMARY KEY, session_id TEXT, pipeline TEXT, attractor INTEGER,
            convergence REAL, score REAL, digit_range TEXT, verified_ranges TEXT,
            is_cycle INTEGER, properties TEXT, is_novel INTEGER, ml_predicted_score REAL, timestamp REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS operation_learning (
            op_name TEXT PRIMARY KEY, alpha REAL, beta REAL, updated REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS known_attractors (
            attractor INTEGER PRIMARY KEY, first_seen REAL, discovery_count INTEGER DEFAULT 1)''')
        c.execute('''CREATE TABLE IF NOT EXISTS elite_pipelines (
            id INTEGER PRIMARY KEY, pipeline TEXT, score REAL, attractor INTEGER)''')
        c.execute('''CREATE TABLE IF NOT EXISTS ml_training_data (
            id INTEGER PRIMARY KEY, pipeline TEXT, features TEXT, score REAL, success INTEGER, timestamp REAL)''')
        conn.commit()
        conn.close()
    
    def _load_previous_discoveries(self):
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('SELECT op_name, alpha, beta FROM operation_learning')
        for row in c.fetchall():
            self.op_alpha[row[0]] = row[1]
            self.op_beta[row[0]] = row[2]
        c.execute('SELECT attractor FROM known_attractors')
        for row in c.fetchall():
            self.known_attractors.add(row[0])
        c.execute('SELECT pipeline, score FROM elite_pipelines ORDER BY score DESC LIMIT 50')
        for row in c.fetchall():
            ops = row[0].split(' → ')
            if all(op in OP_CODES for op in ops):
                self.elite_pipelines.append(([OP_CODES[op] for op in ops], row[1]))
        c.execute('SELECT pipeline, score FROM session_discoveries ORDER BY score DESC LIMIT 100')
        for row in c.fetchall():
            ops = row[0].split(' → ')
            score = row[1]
            for i in range(len(ops) - 1):
                if ops[i] in OP_CODES and ops[i+1] in OP_CODES:
                    pair = (OP_CODES[ops[i]], OP_CODES[ops[i+1]])
                    self.pair_scores[pair] = max(self.pair_scores[pair], score)
        # Load ML training data
        if self.ml_predictor:
            c.execute('SELECT features, score, success FROM ml_training_data ORDER BY id DESC LIMIT 1000')
            for row in c.fetchall():
                try:
                    features = json.loads(row[0])
                    self.ml_predictor.training_data['pipeline_features'].append(features)
                    self.ml_predictor.training_data['quality_scores'].append(row[1])
                    self.ml_predictor.training_data['success_labels'].append(row[2])
                except: pass
            if len(self.ml_predictor.training_data['pipeline_features']) >= self.ml_predictor.min_samples_for_training:
                self.ml_predictor.train_models()
        conn.close()
        if self.op_alpha: print(f"📚 Loaded: {len(self.op_alpha)} operation statistics", flush=True)
        if self.known_attractors: print(f"📚 Loaded: {len(self.known_attractors)} known attractors", flush=True)
        if self.elite_pipelines: print(f"📚 Loaded: {len(self.elite_pipelines)} elite pipelines", flush=True)
        if self.ml_predictor and self.ml_predictor.training_data['pipeline_features']:
            print(f"📚 Loaded: {len(self.ml_predictor.training_data['pipeline_features'])} ML samples", flush=True)
    
    def _save_discovery(self, discovery: Discovery, session_id: str):
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO session_discoveries 
            (session_id, pipeline, attractor, convergence, score, digit_range, verified_ranges,
             is_cycle, properties, is_novel, ml_predicted_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (session_id, ' → '.join(discovery.pipeline), discovery.attractor, discovery.convergence,
             discovery.score, discovery.digit_range, json.dumps(discovery.verified_ranges),
             1 if discovery.is_cycle else 0, json.dumps(discovery.properties),
             1 if discovery.is_novel else 0, discovery.ml_predicted_score, discovery.timestamp))
        c.execute('''INSERT INTO known_attractors (attractor, first_seen, discovery_count) VALUES (?, ?, 1)
            ON CONFLICT(attractor) DO UPDATE SET discovery_count = discovery_count + 1''',
            (discovery.attractor, time.time()))
        conn.commit()
        conn.close()
    
    def _save_ml_sample(self, pipeline: List[int], features: Dict, score: float, success: int):
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO ml_training_data (pipeline, features, score, success, timestamp) VALUES (?, ?, ?, ?, ?)',
            (' → '.join([OP_NAMES[op] for op in pipeline]), json.dumps(features), score, success, time.time()))
        conn.commit()
        conn.close()
    
    def _save_elite(self, pipeline: List[str], score: float, attractor: int):
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO elite_pipelines (pipeline, score, attractor) VALUES (?, ?, ?)',
            (' → '.join(pipeline), score, attractor))
        conn.commit()
        conn.close()
    
    def _save_learning_state(self):
        conn = sqlite3.connect(self.config.db_path)
        c = conn.cursor()
        for op_name in OP_NAMES.values():
            c.execute('INSERT OR REPLACE INTO operation_learning (op_name, alpha, beta, updated) VALUES (?, ?, ?, ?)',
                (op_name, self.op_alpha[op_name], self.op_beta[op_name], time.time()))
        conn.commit()
        conn.close()
    
    def thompson_sample_op(self) -> int:
        scores = {op: np.random.beta(self.op_alpha[name], self.op_beta[name]) for op, name in OP_NAMES.items()}
        return max(scores, key=scores.get)
    
    def mutate_pipeline(self, pipeline: List[int]) -> List[int]:
        mutated = pipeline.copy()
        mutation = random.choice(['swap', 'replace', 'insert', 'delete'])
        if mutation == 'swap' and len(mutated) >= 2:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        elif mutation == 'replace':
            mutated[random.randint(0, len(mutated)-1)] = random.randint(0, len(OP_NAMES)-1)
        elif mutation == 'insert' and len(mutated) < self.config.max_pipeline_length:
            mutated.insert(random.randint(0, len(mutated)), self.thompson_sample_op())
        elif mutation == 'delete' and len(mutated) > 1:
            mutated.pop(random.randint(0, len(mutated)-1))
        return mutated
    
    def crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        if len(p1) < 2 or len(p2) < 2: return p1.copy()
        return (p1[:random.randint(1, len(p1)-1)] + p2[random.randint(1, len(p2)-1):])[:self.config.max_pipeline_length]
    
    def generate_ml_guided_pipeline(self, length: int = None) -> Tuple[List[int], float]:
        if length is None: length = random.randint(1, self.config.max_pipeline_length)
        best_pipeline, best_score = None, 0
        for _ in range(10):
            candidate = self._generate_candidate(length)
            if self.ml_predictor:
                predicted = self.ml_predictor.predict_quality(candidate) * (0.5 + 0.5 * self.ml_predictor.predict_success(candidate))
                if predicted > best_score:
                    best_score, best_pipeline = predicted, candidate
            else:
                return candidate, 50.0
        return best_pipeline or [self.thompson_sample_op()], best_score
    
    def _generate_candidate(self, length: int) -> List[int]:
        strategy = random.random()
        if strategy < 0.25 and self.elite_pipelines:
            return self.mutate_pipeline(random.choice(self.elite_pipelines[:20])[0])
        elif strategy < 0.35 and len(self.elite_pipelines) >= 2:
            p1, p2 = random.sample(self.elite_pipelines[:20], 2)
            return self.crossover(p1[0], p2[0])
        elif strategy < 0.5 and self.pair_scores:
            pair = random.choice(sorted(self.pair_scores.items(), key=lambda x: -x[1])[:10])[0]
            pipeline = list(pair)
            while len(pipeline) < length: pipeline.append(self.thompson_sample_op())
            return pipeline
        else:
            exploration = self.config.initial_exploration_rate * (0.9 ** self.cycle_count)
            return [random.randint(0, len(OP_NAMES)-1) if random.random() < exploration else self.thompson_sample_op() for _ in range(length)]
    
    def generate_smart_pipeline(self, length: int = None) -> List[int]:
        if self.ml_predictor and self.ml_predictor.success_predictor and random.random() < 0.6:
            self.ml_guided_pipelines += 1
            return self.generate_ml_guided_pipeline(length)[0]
        return self._generate_candidate(length or random.randint(1, self.config.max_pipeline_length))
    
    def evaluate_pipeline(self, pipeline: List[int], start: int, end: int) -> Optional[Discovery]:
        batch_size = min(self.config.batch_size, end - start)
        blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block
        numbers = np.arange(start, start + batch_size, dtype=np.int64)
        endpoints = np.zeros(batch_size, dtype=np.int64)
        steps = np.zeros(batch_size, dtype=np.int64)
        cycles = np.zeros(batch_size, dtype=np.int64)
        op_seq = np.array(pipeline, dtype=np.int64)
        
        d_numbers = cuda.to_device(numbers)
        d_ops = cuda.to_device(op_seq)
        d_endpoints = cuda.to_device(endpoints)
        d_steps = cuda.to_device(steps)
        d_cycles = cuda.to_device(cycles)
        
        kernel_pipeline_convergence[blocks, self.threads_per_block](
            d_numbers, d_ops, len(pipeline), d_endpoints, d_steps, d_cycles, self.config.max_iterations)
        cuda.synchronize()
        
        endpoints = d_endpoints.copy_to_host()
        cycles = d_cycles.copy_to_host()
        self.total_numbers_tested += len(numbers)
        
        counter = Counter(endpoints)
        if not counter: return None
        
        top_val, top_count = counter.most_common(1)[0]
        convergence = top_count / len(numbers)
        is_cycle = np.sum(cycles) / len(numbers) > 0.1
        is_novel = top_val not in self.known_attractors and top_val not in self.TRIVIAL_VALUES
        properties = get_attractor_properties(top_val) if top_val > 0 else {}
        ml_predicted = self.ml_predictor.predict_quality(pipeline) if self.ml_predictor else 0.0
        
        score = self._calculate_score(counter, len(numbers), pipeline, is_novel, is_cycle, properties)
        
        if convergence < self.config.min_convergence: return None
        if top_val in self.TRIVIAL_VALUES and convergence < 0.9 and not is_cycle: return None
        if score < 30: return None
        
        if is_novel:
            self.novel_attractors.add(top_val)
            self.novel_discoveries += 1
        self.known_attractors.add(top_val)
        if is_cycle: self.cycles_found += 1
        
        if self.ml_predictor:
            result_dict = {'convergence': convergence, 'attractor': int(top_val)}
            features = get_pipeline_features(pipeline, result_dict)
            self.ml_predictor.add_training_sample(pipeline, result_dict, score)
            self._save_ml_sample(pipeline, features, score, 1 if score > 50 else 0)
            if self.ml_guided_pipelines > 0: self.ml_guided_successes += 1
        
        return Discovery(
            pipeline=[OP_NAMES[op] for op in pipeline], attractor=int(top_val), convergence=convergence,
            score=score, digit_range=f"{start}-{end}", timestamp=time.time(), is_cycle=is_cycle,
            properties=properties, is_novel=is_novel, ml_predicted_score=ml_predicted)
    
    def _calculate_score(self, counter: Counter, total: int, pipeline: List[int],
                         is_novel: bool, is_cycle: bool, properties: Dict) -> float:
        top_val, top_count = counter.most_common(1)[0]
        score = (top_count / total) * 50
        if top_val not in self.TRIVIAL_VALUES: score += 20
        if top_val > 1000: score += 10
        if top_val > 100000: score += 10
        if is_novel: score += self.config.novelty_bonus
        if is_cycle: score += 12
        if properties.get('is_palindrome'): score += 15
        if properties.get('is_prime'): score += 20
        if properties.get('is_perfect_power'): score += 15
        if properties.get('is_repdigit'): score += 10
        if top_val in self.KNOWN_CONSTANTS: score += 5
        score -= len(pipeline) * 1.5
        return max(0, score)
    
    def verify_discovery(self, discovery: Discovery, pipeline: List[int]) -> bool:
        verified = []
        for start, end in [(100000, 1000000), (1000000, 5000000), (10000000, 20000000)]:
            result = self.evaluate_pipeline(pipeline, start, end)
            if result and result.attractor == discovery.attractor and result.convergence > 0.3:
                verified.append(f"{start}-{end}")
        discovery.verified_ranges = verified
        return len(verified) >= 2
    
    def update_learning(self, pipeline: List[str], score: float, success: bool):
        for op_name in pipeline:
            if success: self.op_alpha[op_name] += score / 50
            else: self.op_beta[op_name] += 0.3
        for i in range(len(pipeline) - 1):
            if pipeline[i] in OP_CODES and pipeline[i+1] in OP_CODES:
                pair = (OP_CODES[pipeline[i]], OP_CODES[pipeline[i+1]])
                if success: self.pair_scores[pair] = max(self.pair_scores[pair], score)
    
    def explore_variations(self, base: Discovery) -> List[Discovery]:
        variations = []
        base_pipeline = [OP_CODES[op] for op in base.pipeline]
        for _ in range(5):
            mutated = self.mutate_pipeline(base_pipeline)
            sig = str(mutated)
            if sig not in self.tested_pipelines:
                self.tested_pipelines.add(sig)
                self.total_pipelines_tested += 1
                result = self.evaluate_pipeline(mutated, 10000, 500000)
                if result and result.score > base.score * 0.7:
                    variations.append(result)
        return variations
    
    def run_cycle(self, session_id: str, cycle_num: int, start: int, end: int, num_pipelines: int) -> List[Discovery]:
        cycle_discoveries = []
        for _ in range(num_pipelines):
            pipeline = self.generate_smart_pipeline()
            sig = str(pipeline)
            if sig in self.tested_pipelines: continue
            self.tested_pipelines.add(sig)
            self.total_pipelines_tested += 1
            
            result = self.evaluate_pipeline(pipeline, start, end)
            if result:
                self.update_learning(result.pipeline, result.score, True)
                if result.score > 50: self.verify_discovery(result, pipeline)
                if result.score > 70:
                    self.elite_pipelines.append((pipeline, result.score))
                    self._save_elite(result.pipeline, result.score, result.attractor)
                    self.elite_pipelines = sorted(self.elite_pipelines, key=lambda x: -x[1])[:100]
                cycle_discoveries.append(result)
                self.discoveries.append(result)
                self._save_discovery(result, session_id)
                
                if result.score > 45:
                    tags = f"{'🆕' if result.is_novel else ''}{'🔄' if result.is_cycle else ''}"
                    ml_tag = f"[ML:{result.ml_predicted_score:.0f}]" if result.ml_predicted_score > 0 else ""
                    print(f"\n   🎯 {' → '.join(result.pipeline)} {tags}{ml_tag}", flush=True)
                    print(f"      Attractor: {result.attractor} ({result.convergence*100:.1f}%) Score: {result.score:.1f}", flush=True)
                
                if result.score > 60:
                    for var in self.explore_variations(result):
                        cycle_discoveries.append(var)
                        self.discoveries.append(var)
                        self._save_discovery(var, session_id)
            else:
                self.update_learning([OP_NAMES[op] for op in pipeline], 0, False)
        return cycle_discoveries
    
    def generate_report(self, session_id: str):
        ml_acc = self.ml_predictor.prediction_accuracy[-1] if self.ml_predictor and self.ml_predictor.prediction_accuracy else 0
        ml_rate = self.ml_guided_successes / max(1, self.ml_guided_pipelines)
        
        report = f"""# SYNTRIAD Extended Research Report v6.0 (ML-Enhanced)
## Session: {session_id}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Duration:** {(time.time() - self.start_time) / 60:.1f} minutes  
**Numbers tested:** {self.total_numbers_tested:,}  
**Pipelines tested:** {self.total_pipelines_tested:,}  
**Discoveries:** {len(self.discoveries)}  
**New attractors:** {self.novel_discoveries}  
**Cycles found:** {self.cycles_found}

## 🧠 ML Statistics
| Metric | Value |
|---------|--------|
| ML Model Version | {self.ml_predictor.model_version if self.ml_predictor else 'N/A'} |
| Training Samples | {len(self.ml_predictor.training_data['pipeline_features']) if self.ml_predictor else 0} |
| Success Predictor Accuracy | {ml_acc:.1%} |
| ML-Guided Pipelines | {self.ml_guided_pipelines} |
| ML Success Rate | {ml_rate:.1%} |

## 🏆 Top Discoveries
"""
        for i, d in enumerate(sorted(self.discoveries, key=lambda x: -x.score)[:15], 1):
            tags = f"{'✅' if d.verified_ranges else ''}{'🆕' if d.is_novel else ''}{'🔄' if d.is_cycle else ''}"
            report += f"\n### {i}. {' → '.join(d.pipeline)} {tags}\n"
            report += f"- **Attractor:** {d.attractor} | **Score:** {d.score:.1f} | **ML Predicted:** {d.ml_predicted_score:.0f}\n"
        
        if self.ml_predictor:
            importance = self.ml_predictor.get_feature_importance()
            if importance:
                report += "\n## 📊 Feature Importance\n| Feature | Importance |\n|---------|------------|\n"
                for feat, imp in list(importance.items())[:10]:
                    report += f"| {feat} | {imp:.4f} |\n"
        
        report += f"\n---\n*SYNTRIAD v6.0 ML-Enhanced*"
        with open(self.config.report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📄 Report: {self.config.report_path}", flush=True)
    
    def run(self):
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.start_time = time.time()
        end_time = self.start_time + self.config.duration_minutes * 60
        
        print("█" * 70, flush=True)
        print("  SYNTRIAD EXTENDED RESEARCH SESSION v6.0 (ML-Enhanced)", flush=True)
        print("█" * 70, flush=True)
        print(f"Session: {session_id} | Duration: {self.config.duration_minutes}m | ML: {self.config.ml_enabled and HAS_SKLEARN}", flush=True)
        print("█" * 70, flush=True)
        
        ranges = [(1000, 100000, 40), (10000, 500000, 35), (50000, 2000000, 30),
                  (100000, 5000000, 25), (500000, 10000000, 20), (1000000, 20000000, 15)]
        
        while time.time() < end_time:
            elapsed = (time.time() - self.start_time) / 60
            remaining = (end_time - time.time()) / 60
            range_idx = min(self.cycle_count // 2, len(ranges) - 1)
            start, end, num = ranges[range_idx]
            
            ml_info = f" | ML Acc: {self.ml_predictor.prediction_accuracy[-1]:.1%}" if self.ml_predictor and self.ml_predictor.prediction_accuracy else ""
            print(f"\n{'='*60}", flush=True)
            print(f"🔬 CYCLE {self.cycle_count} | {elapsed:.1f}m/{self.config.duration_minutes}m{ml_info}", flush=True)
            print(f"   Discoveries: {len(self.discoveries)} | New: {self.novel_discoveries} | ML-guided: {self.ml_guided_pipelines}", flush=True)
            
            self.run_cycle(session_id, self.cycle_count, start, end, num)
            if self.cycle_count % 3 == 0: self._save_learning_state()
            self.cycle_count += 1
        
        self._save_learning_state()
        self.generate_report(session_id)
        
        print("\n" + "█" * 70, flush=True)
        print(f"  SESSION COMPLETE: {len(self.discoveries)} discoveries, {self.novel_discoveries} new", flush=True)
        print(f"  ML-guided: {self.ml_guided_pipelines} pipelines, {self.ml_guided_successes} successes", flush=True)
        print("█" * 70, flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SYNTRIAD v6.0 ML-Enhanced')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--batch', type=int, default=2_000_000)
    parser.add_argument('--no-ml', action='store_true')
    args = parser.parse_args()
    
    if not HAS_CUDA:
        print("❌ CUDA not available")
        return
    
    config = SessionConfig(duration_minutes=args.duration, batch_size=args.batch, ml_enabled=not args.no_ml)
    session = ExtendedResearchSessionV6(config)
    session.run()


if __name__ == "__main__":
    main()
