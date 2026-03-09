#!/usr/bin/env python3
"""
SYNTRIAD Meta-Learning Symmetry Discovery Engine v5.0
======================================================

A self-adapting mathematical agent that:
1. Represents symmetries as first-class objects
2. Learns operator embeddings
3. Dynamically adjusts search strategy (meta-learning)
4. Builds a Theory Graph with relations between discoveries
5. Measures entropy/compression as a fundamental metric
6. Reflects on and improves itself

This is no longer a script - this is an experimental mathematical agent.

Hardware: RTX 4000 Ada, 32-core i9, 64GB RAM
"""

import numpy as np
import time
import json
import sqlite3
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Callable, Any
from collections import Counter, defaultdict
from pathlib import Path
from enum import Enum
import itertools
import hashlib
from abc import ABC, abstractmethod

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
# INFORMATION THEORY UTILITIES
# =============================================================================

def digit_entropy(n: int) -> float:
    """Compute Shannon entropy of digit distribution."""
    if n == 0:
        return 0.0
    digits = list(str(abs(n)))
    freqs = Counter(digits)
    total = len(digits)
    probs = [v / total for v in freqs.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def digit_variance(n: int) -> float:
    """Compute variance of digits."""
    if n == 0:
        return 0.0
    digits = [int(d) for d in str(abs(n))]
    if len(digits) < 2:
        return 0.0
    mean = sum(digits) / len(digits)
    return sum((d - mean) ** 2 for d in digits) / len(digits)


def kolmogorov_complexity_estimate(n: int) -> float:
    """Estimate Kolmogorov complexity via compression."""
    s = str(n)
    # Simple estimate: unique substrings / length
    substrings = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            substrings.add(s[i:j])
    return len(substrings) / (len(s) ** 2) if len(s) > 0 else 0


# =============================================================================
# SYMMETRY AS FIRST-CLASS OBJECT
# =============================================================================

@dataclass
class SymmetryProfile:
    """Formal representation of symmetry properties."""

    # Permutation invariances
    digit_permutation_invariant: bool = False
    reversal_invariant: bool = False
    complement_invariant: bool = False  # f(n) == f(complement_9(n))
    
    # Modular invariances
    mod_invariants: Dict[int, bool] = field(default_factory=dict)  # mod 3, 9, 11
    
    # Structural properties
    length_preserving: bool = False
    monotonic_reducing: bool = False
    parity_preserving: bool = False
    
    # Information-theoretic properties
    entropy_reduction_rate: float = 0.0
    variance_change_rate: float = 0.0
    compression_ratio: float = 0.0
    
    # Attractor properties
    creates_fixed_point: bool = False
    creates_cycle: bool = False
    cycle_length: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            float(self.digit_permutation_invariant),
            float(self.reversal_invariant),
            float(self.complement_invariant),
            float(self.mod_invariants.get(3, False)),
            float(self.mod_invariants.get(9, False)),
            float(self.mod_invariants.get(11, False)),
            float(self.length_preserving),
            float(self.monotonic_reducing),
            float(self.parity_preserving),
            self.entropy_reduction_rate,
            self.variance_change_rate,
            self.compression_ratio,
            float(self.creates_fixed_point),
            float(self.creates_cycle),
            min(self.cycle_length / 10.0, 1.0)
        ])
    
    def similarity(self, other: 'SymmetryProfile') -> float:
        """Compute cosine similarity with another profile."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


class SymmetryAnalyzer:
    """Analyzes symmetry properties of operators and pipelines."""
    
    def __init__(self, operations: Dict[str, Callable]):
        self.ops = operations
    
    def analyze_pipeline(self, pipeline: Tuple[str, ...], 
                         sample_size: int = 1000,
                         domain: Tuple[int, int] = (100, 99999)) -> SymmetryProfile:
        """Analyze symmetry properties of a pipeline."""
        
        profile = SymmetryProfile()
        
        # Sample numbers
        numbers = random.sample(range(domain[0], domain[1] + 1), 
                               min(sample_size, domain[1] - domain[0]))
        
        # Compute pipeline function
        def apply_pipeline(n: int) -> int:
            for op_name in pipeline:
                if op_name in self.ops:
                    n = self.ops[op_name](n)
                    if n > 10**15:
                        return -1
            return n
        
        # Test reversal invariance: f(n) == f(reverse(n))
        reversal_matches = 0
        for n in numbers[:200]:
            rev_n = int(str(n)[::-1])
            if apply_pipeline(n) == apply_pipeline(rev_n):
                reversal_matches += 1
        profile.reversal_invariant = reversal_matches > 180
        
        # Test digit permutation invariance (sample)
        perm_matches = 0
        for n in numbers[:100]:
            digits = list(str(n))
            random.shuffle(digits)
            perm_n = int(''.join(digits).lstrip('0') or '0')
            if apply_pipeline(n) == apply_pipeline(perm_n):
                perm_matches += 1
        profile.digit_permutation_invariant = perm_matches > 90
        
        # Test modular invariance
        for mod in [3, 9, 11]:
            mod_matches = 0
            for n in numbers[:200]:
                result = apply_pipeline(n)
                if result >= 0 and n % mod == result % mod:
                    mod_matches += 1
            profile.mod_invariants[mod] = mod_matches > 180
        
        # Test length preservation
        length_preserved = 0
        for n in numbers[:200]:
            result = apply_pipeline(n)
            if result > 0 and len(str(n)) == len(str(result)):
                length_preserved += 1
        profile.length_preserving = length_preserved > 180
        
        # Test monotonic reduction
        reductions = 0
        for n in numbers[:200]:
            result = apply_pipeline(n)
            if 0 < result < n:
                reductions += 1
        profile.monotonic_reducing = reductions > 150
        
        # Test parity preservation
        parity_preserved = 0
        for n in numbers[:200]:
            result = apply_pipeline(n)
            if result >= 0 and n % 2 == result % 2:
                parity_preserved += 1
        profile.parity_preserving = parity_preserved > 180
        
        # Entropy analysis
        entropy_changes = []
        variance_changes = []
        for n in numbers[:200]:
            result = apply_pipeline(n)
            if result > 0:
                e_before = digit_entropy(n)
                e_after = digit_entropy(result)
                entropy_changes.append(e_before - e_after)
                
                v_before = digit_variance(n)
                v_after = digit_variance(result)
                if v_before > 0:
                    variance_changes.append((v_before - v_after) / v_before)
        
        if entropy_changes:
            profile.entropy_reduction_rate = np.mean(entropy_changes)
        if variance_changes:
            profile.variance_change_rate = np.mean(variance_changes)
        
        # Fixed point / cycle detection
        fixed_points = 0
        cycles = 0
        cycle_lengths = []
        
        for n in numbers[:100]:
            seen = {n: 0}
            current = n
            for step in range(50):
                current = apply_pipeline(current)
                if current < 0:
                    break
                if current in seen:
                    cycle_len = step + 1 - seen[current]
                    if cycle_len == 1:
                        fixed_points += 1
                    else:
                        cycles += 1
                        cycle_lengths.append(cycle_len)
                    break
                seen[current] = step + 1
        
        profile.creates_fixed_point = fixed_points > 80
        profile.creates_cycle = cycles > 10
        if cycle_lengths:
            profile.cycle_length = int(np.median(cycle_lengths))
        
        return profile


# =============================================================================
# OPERATOR EMBEDDING LAYER
# =============================================================================

@dataclass
class OperatorFeatures:
    """Feature vector for an operator."""
    name: str
    
    # Structural properties
    digit_reordering: float = 0.0  # 0-1: how much it reorders digits
    length_preserving: float = 0.0  # 0-1: preserves length
    monotonic_reducing: float = 0.0  # 0-1: reduces value

    # Algebraic properties
    parity_preserving: float = 0.0
    mod9_preserving: float = 0.0
    mod11_preserving: float = 0.0
    
    # Symmetry properties
    reversal_symmetric: float = 0.0
    creates_symmetry: float = 0.0
    
    # Information properties
    entropy_effect: float = 0.0  # negative = reduces, positive = increases
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.digit_reordering,
            self.length_preserving,
            self.monotonic_reducing,
            self.parity_preserving,
            self.mod9_preserving,
            self.mod11_preserving,
            self.reversal_symmetric,
            self.creates_symmetry,
            self.entropy_effect
        ])


class OperatorEmbedding:
    """Learns and manages operator embeddings."""
    
    def __init__(self, operations: Dict[str, Callable]):
        self.ops = operations
        self.embeddings: Dict[str, OperatorFeatures] = {}
        self._compute_embeddings()
    
    def _compute_embeddings(self):
        """Compute embeddings for all operators."""
        for name, op in self.ops.items():
            features = OperatorFeatures(name=name)
            
            # Test on sample numbers
            test_numbers = [random.randint(100, 99999) for _ in range(500)]
            
            length_preserved = 0
            value_reduced = 0
            parity_preserved = 0
            mod9_preserved = 0
            mod11_preserved = 0
            reversal_symmetric = 0
            entropy_changes = []
            
            for n in test_numbers:
                try:
                    result = op(n)
                    if result <= 0 or result > 10**15:
                        continue
                    
                    # Length
                    if len(str(n)) == len(str(result)):
                        length_preserved += 1
                    
                    # Value reduction
                    if result < n:
                        value_reduced += 1
                    
                    # Parity
                    if n % 2 == result % 2:
                        parity_preserved += 1
                    
                    # Mod preservation
                    if n % 9 == result % 9:
                        mod9_preserved += 1
                    if n % 11 == result % 11:
                        mod11_preserved += 1
                    
                    # Reversal symmetry
                    rev_n = int(str(n)[::-1])
                    rev_result = op(rev_n)
                    if result == rev_result:
                        reversal_symmetric += 1
                    
                    # Entropy
                    e_before = digit_entropy(n)
                    e_after = digit_entropy(result)
                    entropy_changes.append(e_after - e_before)
                    
                except:
                    continue
            
            n_valid = len(test_numbers)
            features.length_preserving = length_preserved / n_valid
            features.monotonic_reducing = value_reduced / n_valid
            features.parity_preserving = parity_preserved / n_valid
            features.mod9_preserving = mod9_preserved / n_valid
            features.mod11_preserving = mod11_preserved / n_valid
            features.reversal_symmetric = reversal_symmetric / n_valid
            
            if entropy_changes:
                features.entropy_effect = np.mean(entropy_changes)
            
            # Digit reordering (heuristic based on name)
            if any(x in name for x in ['sort', 'reverse', 'rotate', 'swap']):
                features.digit_reordering = 0.8
            elif any(x in name for x in ['sum', 'product', 'pow']):
                features.digit_reordering = 0.2
            
            self.embeddings[name] = features
    
    def get_embedding(self, name: str) -> Optional[OperatorFeatures]:
        return self.embeddings.get(name)
    
    def pipeline_embedding(self, pipeline: Tuple[str, ...]) -> np.ndarray:
        """Compute combined embedding for pipeline."""
        vectors = []
        for op_name in pipeline:
            if op_name in self.embeddings:
                vectors.append(self.embeddings[op_name].to_vector())
        
        if not vectors:
            return np.zeros(9)
        
        # Combine: mean + std + first + last
        arr = np.array(vectors)
        combined = np.concatenate([
            np.mean(arr, axis=0),
            np.std(arr, axis=0),
            arr[0],
            arr[-1]
        ])
        return combined
    
    def similarity(self, op1: str, op2: str) -> float:
        """Compute similarity between two operators."""
        e1 = self.embeddings.get(op1)
        e2 = self.embeddings.get(op2)
        if not e1 or not e2:
            return 0.0
        
        v1 = e1.to_vector()
        v2 = e2.to_vector()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


# =============================================================================
# THEORY GRAPH MEMORY
# =============================================================================

class NodeType(Enum):
    PIPELINE = "pipeline"
    ATTRACTOR = "attractor"
    SYMMETRY = "symmetry"
    OPERATOR = "operator"
    PROPERTY = "property"
    THEORY = "theory"


class EdgeType(Enum):
    PRODUCES = "produces"
    SHARES_SYMMETRY = "shares_symmetry"
    REFINES = "refines"
    GENERALIZES = "generalizes"
    CONTRADICTS = "contradicts"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"


@dataclass
class TheoryNode:
    """Node in the theory graph."""
    id: str
    type: NodeType
    data: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    confidence: float = 0.5


@dataclass
class TheoryEdge:
    """Edge in the theory graph."""
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)


class TheoryGraph:
    """Knowledge graph for theories and relations."""
    
    def __init__(self):
        self.nodes: Dict[str, TheoryNode] = {}
        self.edges: List[TheoryEdge] = []
        self.theories: List[str] = []  # High-level theories
    
    def add_node(self, node: TheoryNode) -> str:
        self.nodes[node.id] = node
        return node.id
    
    def add_edge(self, edge: TheoryEdge):
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[TheoryNode]:
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[TheoryEdge]:
        return [e for e in self.edges if e.source_id == node_id]
    
    def get_edges_to(self, node_id: str) -> List[TheoryEdge]:
        return [e for e in self.edges if e.target_id == node_id]
    
    def find_similar_nodes(self, node_id: str, threshold: float = 0.7) -> List[str]:
        """Find nodes that have similar_to edges above threshold."""
        similar = []
        for edge in self.edges:
            if edge.type == EdgeType.SIMILAR_TO:
                if edge.source_id == node_id and edge.weight >= threshold:
                    similar.append(edge.target_id)
                elif edge.target_id == node_id and edge.weight >= threshold:
                    similar.append(edge.source_id)
        return similar
    
    def add_pipeline_discovery(self, pipeline: Tuple[str, ...], 
                                attractor: Tuple[int, ...],
                                symmetry_profile: SymmetryProfile,
                                dominance: float):
        """Add a pipeline discovery to the graph."""
        
        # Pipeline node
        pipeline_id = f"pipeline_{hash(pipeline)}"
        pipeline_node = TheoryNode(
            id=pipeline_id,
            type=NodeType.PIPELINE,
            data={"operations": list(pipeline), "dominance": dominance},
            confidence=min(0.99, dominance / 100)
        )
        self.add_node(pipeline_node)
        
        # Attractor node
        attractor_id = f"attractor_{hash(attractor)}"
        attractor_node = TheoryNode(
            id=attractor_id,
            type=NodeType.ATTRACTOR,
            data={"values": list(attractor), "is_fixed_point": len(attractor) == 1},
            confidence=0.95
        )
        self.add_node(attractor_node)
        
        # Edge: pipeline produces attractor
        self.add_edge(TheoryEdge(
            source_id=pipeline_id,
            target_id=attractor_id,
            type=EdgeType.PRODUCES,
            weight=dominance / 100,
            evidence={"dominance": dominance}
        ))
        
        # Symmetry node
        symmetry_id = f"symmetry_{hash(str(symmetry_profile.to_vector().tobytes()))}"
        symmetry_node = TheoryNode(
            id=symmetry_id,
            type=NodeType.SYMMETRY,
            data={
                "reversal_invariant": symmetry_profile.reversal_invariant,
                "mod9_invariant": symmetry_profile.mod_invariants.get(9, False),
                "entropy_reduction": symmetry_profile.entropy_reduction_rate
            },
            confidence=0.8
        )
        self.add_node(symmetry_node)
        
        # Edge: pipeline has symmetry
        self.add_edge(TheoryEdge(
            source_id=pipeline_id,
            target_id=symmetry_id,
            type=EdgeType.CONTAINS,
            weight=1.0
        ))
        
        return pipeline_id
    
    def find_shared_symmetries(self) -> List[Tuple[str, str, str]]:
        """Find pipelines that share symmetries."""
        shared = []
        
        # Group pipelines by symmetry
        symmetry_to_pipelines: Dict[str, List[str]] = defaultdict(list)
        
        for edge in self.edges:
            if edge.type == EdgeType.CONTAINS:
                source = self.get_node(edge.source_id)
                target = self.get_node(edge.target_id)
                if source and target:
                    if source.type == NodeType.PIPELINE and target.type == NodeType.SYMMETRY:
                        symmetry_to_pipelines[edge.target_id].append(edge.source_id)
        
        # Find pairs
        for symmetry_id, pipelines in symmetry_to_pipelines.items():
            if len(pipelines) >= 2:
                for i, p1 in enumerate(pipelines):
                    for p2 in pipelines[i+1:]:
                        shared.append((p1, p2, symmetry_id))
        
        return shared
    
    def generate_theory(self) -> Optional[str]:
        """Generate a high-level theory from the graph."""

        # Search for patterns
        shared = self.find_shared_symmetries()

        if len(shared) >= 3:
            # There are multiple pipelines with shared symmetry
            symmetry_node = self.get_node(shared[0][2])
            if symmetry_node:
                theory = f"THEORY: Pipelines with {symmetry_node.data} tend to share attractor structures"
                
                theory_node = TheoryNode(
                    id=f"theory_{len(self.theories)}",
                    type=NodeType.THEORY,
                    data={"statement": theory, "evidence_count": len(shared)},
                    confidence=min(0.9, 0.5 + len(shared) * 0.1)
                )
                self.add_node(theory_node)
                self.theories.append(theory_node.id)
                
                return theory
        
        return None
    
    def summary(self) -> Dict:
        """Generate summary of the graph."""
        return {
            "total_nodes": len(self.nodes),
            "pipelines": sum(1 for n in self.nodes.values() if n.type == NodeType.PIPELINE),
            "attractors": sum(1 for n in self.nodes.values() if n.type == NodeType.ATTRACTOR),
            "symmetries": sum(1 for n in self.nodes.values() if n.type == NodeType.SYMMETRY),
            "theories": len(self.theories),
            "edges": len(self.edges),
            "shared_symmetries": len(self.find_shared_symmetries())
        }


# =============================================================================
# META-LEARNING SEARCH CONTROLLER
# =============================================================================

@dataclass
class SearchState:
    """Current state of the search strategy."""

    # Weights for score function
    dominance_weight: float = 0.4
    symmetry_weight: float = 0.3
    novelty_weight: float = 0.2
    compression_weight: float = 0.1
    
    # Search parameters
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    exploration_rate: float = 0.5  # vs exploitation
    
    # Statistics
    total_explored: int = 0
    successful_discoveries: int = 0
    confirmed_hypotheses: int = 0
    refuted_hypotheses: int = 0
    
    # Operator biases
    operator_scores: Dict[str, float] = field(default_factory=dict)
    category_biases: Dict[str, float] = field(default_factory=dict)


class MetaLearningController:
    """Self-adapting search controller."""
    
    def __init__(self, operations: List[str]):
        self.ops = operations
        self.state = SearchState()
        self.history: List[Dict] = []
        
        # Operator categories
        self.categories = {
            "reducing": ["digit_sum", "digit_pow2", "digit_pow3", "digit_pow4", "digit_pow5"],
            "transforming": ["reverse", "sort_asc", "sort_desc", "swap_ends", "rotate_left", "rotate_right"],
            "combining": ["kaprekar_step", "truc_1089", "add_reverse", "sub_reverse"],
            "special": ["digit_factorial_sum", "happy_step", "complement_9", "collatz_step"]
        }
        
        # Initialize biases
        for cat in self.categories:
            self.state.category_biases[cat] = 1.0
        for op in operations:
            self.state.operator_scores[op] = 1.0
    
    def compute_score(self, dominance: float, symmetry_score: float,
                      novelty: float, compression: float) -> float:
        """Compute weighted score for a discovery."""
        return (
            self.state.dominance_weight * dominance / 100 +
            self.state.symmetry_weight * symmetry_score +
            self.state.novelty_weight * novelty +
            self.state.compression_weight * compression
        )
    
    def select_operators(self, n: int = 2) -> Tuple[str, ...]:
        """Select operators with bias."""
        
        if random.random() < self.state.exploration_rate:
            # Exploration: random
            return tuple(random.choices(self.ops, k=n))
        else:
            # Exploitation: weighted toward scores
            weights = [self.state.operator_scores.get(op, 1.0) for op in self.ops]
            total = sum(weights)
            probs = [w / total for w in weights]
            return tuple(np.random.choice(self.ops, size=n, p=probs, replace=True))
    
    def select_category_pipeline(self) -> Tuple[str, ...]:
        """Select pipeline based on category biases."""

        # Select 2 categories
        cats = list(self.categories.keys())
        weights = [self.state.category_biases.get(c, 1.0) for c in cats]
        total = sum(weights)
        probs = [w / total for w in weights]
        
        selected_cats = np.random.choice(cats, size=2, p=probs, replace=True)
        
        ops = []
        for cat in selected_cats:
            ops.append(random.choice(self.categories[cat]))
        
        return tuple(ops)
    
    def record_result(self, pipeline: Tuple[str, ...], 
                      dominance: float,
                      symmetry_profile: SymmetryProfile,
                      is_novel: bool):
        """Record result and update biases."""
        
        self.state.total_explored += 1
        
        # Compute scores
        symmetry_score = sum([
            float(symmetry_profile.reversal_invariant),
            float(symmetry_profile.mod_invariants.get(9, False)),
            float(symmetry_profile.creates_fixed_point),
            abs(symmetry_profile.entropy_reduction_rate)
        ]) / 4
        
        novelty = 1.0 if is_novel else 0.0
        compression = max(0, symmetry_profile.entropy_reduction_rate)
        
        score = self.compute_score(dominance, symmetry_score, novelty, compression)
        
        # Update operator scores
        for op in pipeline:
            old_score = self.state.operator_scores.get(op, 1.0)
            # Exponential moving average
            self.state.operator_scores[op] = 0.9 * old_score + 0.1 * (score * 2)
        
        # Update category biases
        for cat, ops in self.categories.items():
            if any(op in ops for op in pipeline):
                old_bias = self.state.category_biases.get(cat, 1.0)
                self.state.category_biases[cat] = 0.9 * old_bias + 0.1 * (score * 2)
        
        # Track success
        if dominance > 90:
            self.state.successful_discoveries += 1
        
        # Save to history
        self.history.append({
            "pipeline": pipeline,
            "dominance": dominance,
            "score": score,
            "symmetry_score": symmetry_score
        })
    
    def adapt_strategy(self):
        """Adapt search strategy based on results."""
        
        if len(self.history) < 10:
            return
        
        recent = self.history[-20:]
        
        # Compute success rate
        success_rate = sum(1 for h in recent if h["dominance"] > 90) / len(recent)
        
        # Adjust exploration rate
        if success_rate < 0.1:
            # Low success: explore more
            self.state.exploration_rate = min(0.8, self.state.exploration_rate + 0.05)
            self.state.mutation_rate = min(0.5, self.state.mutation_rate + 0.05)
        elif success_rate > 0.3:
            # High success: exploit more
            self.state.exploration_rate = max(0.2, self.state.exploration_rate - 0.05)
        
        # Adjust weights based on correlations
        high_dom = [h for h in recent if h["dominance"] > 80]
        if high_dom:
            avg_sym = np.mean([h["symmetry_score"] for h in high_dom])
            if avg_sym > 0.5:
                # Symmetry correlates with success
                self.state.symmetry_weight = min(0.5, self.state.symmetry_weight + 0.02)
                self.state.dominance_weight = max(0.2, self.state.dominance_weight - 0.02)
    
    def get_status(self) -> Dict:
        """Return current status."""
        return {
            "total_explored": self.state.total_explored,
            "successful_discoveries": self.state.successful_discoveries,
            "exploration_rate": self.state.exploration_rate,
            "mutation_rate": self.state.mutation_rate,
            "top_operators": sorted(
                self.state.operator_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "category_biases": self.state.category_biases
        }


# =============================================================================
# SELF-REFLECTION SYSTEM
# =============================================================================

@dataclass
class ReflectionInsight:
    """An insight from self-reflection."""
    type: str
    observation: str
    recommendation: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


class SelfReflectionSystem:
    """Analyzes own performance and generates insights."""
    
    def __init__(self):
        self.insights: List[ReflectionInsight] = []
        self.cycle_stats: List[Dict] = []
    
    def record_cycle(self, stats: Dict):
        """Record statistics from a cycle."""
        self.cycle_stats.append({
            **stats,
            "timestamp": time.time()
        })
    
    def reflect(self, theory_graph: TheoryGraph, 
                search_controller: MetaLearningController) -> List[ReflectionInsight]:
        """Perform self-reflection and generate insights."""
        
        insights = []
        
        if len(self.cycle_stats) < 3:
            return insights
        
        recent = self.cycle_stats[-5:]
        
        # Insight 1: Discovery rate trend
        discovery_rates = [s.get("discoveries", 0) / max(s.get("explored", 1), 1) 
                          for s in recent]
        
        if len(discovery_rates) >= 3:
            trend = discovery_rates[-1] - discovery_rates[0]
            if trend < -0.1:
                insights.append(ReflectionInsight(
                    type="performance_decline",
                    observation=f"Discovery rate declining: {discovery_rates[0]:.2f} → {discovery_rates[-1]:.2f}",
                    recommendation="Increase exploration rate or try new operator combinations",
                    confidence=0.8
                ))
            elif trend > 0.1:
                insights.append(ReflectionInsight(
                    type="performance_improvement",
                    observation=f"Discovery rate improving: {discovery_rates[0]:.2f} → {discovery_rates[-1]:.2f}",
                    recommendation="Current strategy is working, continue exploitation",
                    confidence=0.8
                ))
        
        # Insight 2: Operator effectiveness
        controller_status = search_controller.get_status()
        top_ops = controller_status.get("top_operators", [])
        
        if top_ops:
            best_op = top_ops[0][0]
            worst_ops = [op for op, score in search_controller.state.operator_scores.items() 
                        if score < 0.5]
            
            if worst_ops:
                insights.append(ReflectionInsight(
                    type="operator_analysis",
                    observation=f"Best operator: {best_op}. Underperforming: {worst_ops[:3]}",
                    recommendation=f"Focus on combinations with {best_op}, reduce use of {worst_ops[0]}",
                    confidence=0.7
                ))
        
        # Insight 3: Theory graph analysis
        graph_summary = theory_graph.summary()
        shared_symmetries = graph_summary.get("shared_symmetries", 0)
        
        if shared_symmetries > 5:
            insights.append(ReflectionInsight(
                type="pattern_detected",
                observation=f"Found {shared_symmetries} pipeline pairs sharing symmetries",
                recommendation="Investigate common structural properties of these pipelines",
                confidence=0.85
            ))
        
        # Insight 4: Exploration vs exploitation balance
        exp_rate = search_controller.state.exploration_rate
        success_rate = search_controller.state.successful_discoveries / max(
            search_controller.state.total_explored, 1)
        
        if exp_rate > 0.6 and success_rate > 0.2:
            insights.append(ReflectionInsight(
                type="strategy_adjustment",
                observation=f"High exploration ({exp_rate:.2f}) with good success ({success_rate:.2f})",
                recommendation="Can reduce exploration to exploit successful patterns",
                confidence=0.75
            ))
        
        self.insights.extend(insights)
        return insights
    
    def get_summary(self) -> Dict:
        """Generate summary of insights."""
        return {
            "total_insights": len(self.insights),
            "recent_insights": [
                {"type": i.type, "observation": i.observation[:50]}
                for i in self.insights[-5:]
            ],
            "cycles_analyzed": len(self.cycle_stats)
        }


# =============================================================================
# DIGIT OPERATIONS
# =============================================================================

class DigitOp:
    @staticmethod
    def reverse(n: int) -> int:
        return int(str(abs(n))[::-1]) if n != 0 else 0
    
    @staticmethod
    def digit_sum(n: int) -> int:
        return sum(int(d) for d in str(abs(n)))
    
    @staticmethod
    def digit_product(n: int) -> int:
        result = 1
        for d in str(abs(n)):
            if int(d) > 0:
                result *= int(d)
        return result
    
    @staticmethod
    def digit_pow2(n: int) -> int:
        return sum(int(d)**2 for d in str(abs(n)))
    
    @staticmethod
    def digit_pow3(n: int) -> int:
        return sum(int(d)**3 for d in str(abs(n)))
    
    @staticmethod
    def digit_pow4(n: int) -> int:
        return sum(int(d)**4 for d in str(abs(n)))
    
    @staticmethod
    def digit_pow5(n: int) -> int:
        return sum(int(d)**5 for d in str(abs(n)))
    
    @staticmethod
    def sort_asc(n: int) -> int:
        s = ''.join(sorted(str(abs(n)))).lstrip('0')
        return int(s) if s else 0
    
    @staticmethod
    def sort_desc(n: int) -> int:
        return int(''.join(sorted(str(abs(n)), reverse=True)))
    
    @staticmethod
    def kaprekar_step(n: int) -> int:
        return DigitOp.sort_desc(n) - DigitOp.sort_asc(n)
    
    @staticmethod
    def truc_1089(n: int) -> int:
        if n <= 0:
            return 0
        rev = DigitOp.reverse(n)
        diff = abs(n - rev)
        if diff == 0:
            return 0
        return diff + DigitOp.reverse(diff)
    
    @staticmethod
    def swap_ends(n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        return int((s[-1] + s[1:-1] + s[0]).lstrip('0') or '0')
    
    @staticmethod
    def complement_9(n: int) -> int:
        return int(''.join(str(9 - int(d)) for d in str(abs(n))).lstrip('0') or '0')
    
    @staticmethod
    def add_reverse(n: int) -> int:
        return abs(n) + DigitOp.reverse(n)
    
    @staticmethod
    def sub_reverse(n: int) -> int:
        return abs(abs(n) - DigitOp.reverse(n))
    
    @staticmethod
    def digit_factorial_sum(n: int) -> int:
        factorials = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
        return sum(factorials[int(d)] for d in str(abs(n)))
    
    @staticmethod
    def happy_step(n: int) -> int:
        return DigitOp.digit_pow2(n)
    
    @staticmethod
    def collatz_step(n: int) -> int:
        if n <= 0:
            return 0
        return n // 2 if n % 2 == 0 else 3 * n + 1
    
    @staticmethod
    def rotate_left(n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        return int((s[1:] + s[0]).lstrip('0') or '0')
    
    @staticmethod
    def rotate_right(n: int) -> int:
        s = str(abs(n))
        if len(s) <= 1:
            return n
        return int((s[-1] + s[:-1]).lstrip('0') or '0')


OPERATIONS: Dict[str, Callable[[int], int]] = {
    'reverse': DigitOp.reverse,
    'digit_sum': DigitOp.digit_sum,
    'digit_product': DigitOp.digit_product,
    'digit_pow2': DigitOp.digit_pow2,
    'digit_pow3': DigitOp.digit_pow3,
    'digit_pow4': DigitOp.digit_pow4,
    'digit_pow5': DigitOp.digit_pow5,
    'sort_asc': DigitOp.sort_asc,
    'sort_desc': DigitOp.sort_desc,
    'kaprekar_step': DigitOp.kaprekar_step,
    'truc_1089': DigitOp.truc_1089,
    'swap_ends': DigitOp.swap_ends,
    'complement_9': DigitOp.complement_9,
    'add_reverse': DigitOp.add_reverse,
    'sub_reverse': DigitOp.sub_reverse,
    'digit_factorial_sum': DigitOp.digit_factorial_sum,
    'happy_step': DigitOp.happy_step,
    'collatz_step': DigitOp.collatz_step,
    'rotate_left': DigitOp.rotate_left,
    'rotate_right': DigitOp.rotate_right,
}


# =============================================================================
# META-LEARNING SYMMETRY DISCOVERY ENGINE v5.0
# =============================================================================

class MetaSymmetryEngine:
    """The complete meta-learning symmetry discovery engine."""
    
    def __init__(self, db_path: str = "meta_symmetry_v5.db"):
        self.ops = OPERATIONS
        
        # Core components
        self.symmetry_analyzer = SymmetryAnalyzer(self.ops)
        self.operator_embedding = OperatorEmbedding(self.ops)
        self.theory_graph = TheoryGraph()
        self.search_controller = MetaLearningController(list(self.ops.keys()))
        self.reflection_system = SelfReflectionSystem()
        
        # State
        self.iteration = 0
        self.known_attractors: Set[Tuple[int, ...]] = set()
        
        # Database
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS discoveries (
            id INTEGER PRIMARY KEY,
            timestamp REAL,
            pipeline TEXT,
            attractor TEXT,
            dominance REAL,
            symmetry_profile TEXT,
            score REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY,
            timestamp REAL,
            type TEXT,
            observation TEXT,
            recommendation TEXT
        )''')
        conn.commit()
        conn.close()
    
    def apply_pipeline(self, n: int, pipeline: Tuple[str, ...]) -> int:
        """Apply pipeline to number."""
        for op_name in pipeline:
            if op_name in self.ops:
                n = self.ops[op_name](n)
                if n > 10**15:
                    return -1
        return n
    
    def detect_attractor(self, pipeline: Tuple[str, ...], 
                          domain: Tuple[int, int] = (1000, 99999),
                          sample_size: int = 5000) -> Tuple[Optional[Tuple[int, ...]], float]:
        """Detect dominant attractor for pipeline."""
        
        numbers = random.sample(range(domain[0], domain[1] + 1), 
                               min(sample_size, domain[1] - domain[0]))
        
        attractor_counts: Counter = Counter()
        
        for n in numbers:
            seen = {n: 0}
            current = n
            
            for step in range(100):
                current = self.apply_pipeline(current, pipeline)
                if current < 0:
                    break
                if current in seen:
                    # Found cycle
                    cycle_start = seen[current]
                    cycle = tuple(sorted(set(
                        [current] + [k for k, v in seen.items() if v >= cycle_start]
                    )))
                    attractor_counts[cycle] += 1
                    break
                seen[current] = step + 1
        
        if not attractor_counts:
            return None, 0.0
        
        dominant, count = attractor_counts.most_common(1)[0]
        dominance = 100 * count / len(numbers)
        
        return dominant, dominance
    
    def explore_pipeline(self, pipeline: Tuple[str, ...]) -> Dict:
        """Full exploration of a pipeline."""
        
        # Detect attractor
        attractor, dominance = self.detect_attractor(pipeline)
        
        # Analyze symmetry
        symmetry_profile = self.symmetry_analyzer.analyze_pipeline(pipeline)
        
        # Check novelty
        is_novel = attractor not in self.known_attractors if attractor else False
        if attractor and is_novel:
            self.known_attractors.add(attractor)
        
        # Record in search controller
        self.search_controller.record_result(pipeline, dominance, symmetry_profile, is_novel)
        
        # Add to theory graph
        if attractor and dominance > 50:
            self.theory_graph.add_pipeline_discovery(
                pipeline, attractor, symmetry_profile, dominance
            )
        
        return {
            "pipeline": pipeline,
            "attractor": attractor,
            "dominance": dominance,
            "symmetry_profile": symmetry_profile,
            "is_novel": is_novel
        }
    
    def run_discovery_cycle(self, num_pipelines: int = 20) -> Dict:
        """Run one discovery cycle."""
        
        self.iteration += 1
        cycle_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"🧠 META-LEARNING CYCLE {self.iteration}")
        print(f"{'='*70}")
        
        discoveries = []
        novel_count = 0
        high_dominance_count = 0
        
        for i in range(num_pipelines):
            # Generate pipeline using meta-learning
            if random.random() < 0.5:
                pipeline = self.search_controller.select_operators(
                    n=random.randint(2, 4)
                )
            else:
                pipeline = self.search_controller.select_category_pipeline()
            
            # Explore
            result = self.explore_pipeline(pipeline)
            discoveries.append(result)
            
            if result["is_novel"]:
                novel_count += 1
            if result["dominance"] > 90:
                high_dominance_count += 1
            
            # Print significant discoveries
            if result["dominance"] > 80:
                print(f"\n   [{i+1}] {' → '.join(pipeline)}")
                print(f"       Attractor: {result['attractor']}")
                print(f"       Dominance: {result['dominance']:.1f}%")
                print(f"       Novel: {result['is_novel']}")
                
                sp = result["symmetry_profile"]
                print(f"       Symmetry: rev={sp.reversal_invariant}, "
                      f"mod9={sp.mod_invariants.get(9, False)}, "
                      f"entropy_red={sp.entropy_reduction_rate:.3f}")
        
        # Adapt search strategy
        self.search_controller.adapt_strategy()
        
        # Record cycle stats
        cycle_stats = {
            "cycle": self.iteration,
            "explored": num_pipelines,
            "discoveries": high_dominance_count,
            "novel": novel_count,
            "duration": time.time() - cycle_start
        }
        self.reflection_system.record_cycle(cycle_stats)
        
        # Self-reflection
        insights = self.reflection_system.reflect(
            self.theory_graph, self.search_controller
        )
        
        if insights:
            print(f"\n   💡 SELF-REFLECTION INSIGHTS:")
            for insight in insights:
                print(f"      [{insight.type}] {insight.observation[:60]}...")
                print(f"         → {insight.recommendation[:60]}...")
        
        # Try to generate theory
        theory = self.theory_graph.generate_theory()
        if theory:
            print(f"\n   🧬 NEW THEORY GENERATED:")
            print(f"      {theory}")
        
        return cycle_stats
    
    def run_research_session(self, cycles: int = 5, pipelines_per_cycle: int = 20):
        """Run full research session."""
        
        print("█" * 70)
        print("  SYNTRIAD META-LEARNING SYMMETRY DISCOVERY ENGINE v5.0")
        print("  Self-Adapting Mathematical Research Agent")
        print("█" * 70)
        
        session_start = time.time()
        
        # Phase 1: Initial exploration with learning
        print("\n" + "▓" * 70)
        print("  PHASE 1: META-LEARNING EXPLORATION")
        print("▓" * 70)
        
        for cycle in range(cycles):
            self.run_discovery_cycle(num_pipelines=pipelines_per_cycle)
        
        # Phase 2: Report
        print("\n" + "▓" * 70)
        print("  PHASE 2: ANALYSIS & REPORTING")
        print("▓" * 70)
        
        # Search controller status
        print("\n📊 SEARCH CONTROLLER STATUS:")
        status = self.search_controller.get_status()
        print(f"   Total explored: {status['total_explored']}")
        print(f"   Successful discoveries: {status['successful_discoveries']}")
        print(f"   Exploration rate: {status['exploration_rate']:.2f}")
        print(f"   Top operators: {status['top_operators'][:5]}")
        
        # Theory graph summary
        print("\n🗺️ THEORY GRAPH SUMMARY:")
        graph_summary = self.theory_graph.summary()
        print(f"   Nodes: {graph_summary['total_nodes']}")
        print(f"   Pipelines: {graph_summary['pipelines']}")
        print(f"   Attractors: {graph_summary['attractors']}")
        print(f"   Symmetries: {graph_summary['symmetries']}")
        print(f"   Theories: {graph_summary['theories']}")
        print(f"   Shared symmetries: {graph_summary['shared_symmetries']}")
        
        # Operator embeddings analysis
        print("\n🧬 OPERATOR EMBEDDING ANALYSIS:")
        for op_name in ['digit_pow4', 'truc_1089', 'kaprekar_step', 'sort_desc']:
            emb = self.operator_embedding.get_embedding(op_name)
            if emb:
                print(f"   {op_name}:")
                print(f"      length_preserving: {emb.length_preserving:.2f}")
                print(f"      monotonic_reducing: {emb.monotonic_reducing:.2f}")
                print(f"      mod9_preserving: {emb.mod9_preserving:.2f}")
                print(f"      entropy_effect: {emb.entropy_effect:.3f}")
        
        # Self-reflection summary
        print("\n💭 SELF-REFLECTION SUMMARY:")
        reflection_summary = self.reflection_system.get_summary()
        print(f"   Total insights: {reflection_summary['total_insights']}")
        print(f"   Cycles analyzed: {reflection_summary['cycles_analyzed']}")
        for insight in reflection_summary['recent_insights']:
            print(f"   - [{insight['type']}] {insight['observation']}")
        
        # Final summary
        session_duration = time.time() - session_start
        
        print("\n" + "█" * 70)
        print("  SESSION COMPLETE")
        print("█" * 70)
        print(f"\n   Duration: {session_duration:.1f}s")
        print(f"   Cycles: {self.iteration}")
        print(f"   Pipelines explored: {status['total_explored']}")
        print(f"   Novel attractors: {len(self.known_attractors)}")
        print(f"   Theories generated: {graph_summary['theories']}")
        
        # Top discoveries
        print("\n   📋 TOP DISCOVERIES (by dominance):")
        top_pipelines = sorted(
            self.search_controller.history,
            key=lambda x: x['dominance'],
            reverse=True
        )[:5]
        for h in top_pipelines:
            print(f"      {' → '.join(h['pipeline'])}: {h['dominance']:.1f}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    engine = MetaSymmetryEngine()
    engine.run_research_session(cycles=5, pipelines_per_cycle=25)
