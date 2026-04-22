#!/usr/bin/env python3
"""
SYNTRIAD v6.0 - Scoring Engine v2
=================================

Revised scoring algorithm that filters trivial attractors and rewards
mathematically interesting patterns.

Features:
- Trivial attractor filtering (0-9)
- Trivial cycle detection and filtering
- Property-based bonus scoring
- ML feature importance integration
- Configurable scoring weights

Requirements:
- INTL-01: Implement revised scoring algorithm that filters trivial attractors (0-9)
- INTL-02: Assign 0 score to attractors < 10 and exclude from top results
- INTL-03: Filter cycles of length 1-2 containing only trivial values
- INTL-10: Verify <5% trivial results in top-100 discoveries

Author: SYNTRIAD Research
Created: 2026-01-27
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import Counter
from enum import Enum, auto


class AttractorCategory(Enum):
    """Categories of attractor values."""
    TRIVIAL = auto()          # 0-9
    LOW_VALUE = auto()        # 10-99
    MEDIUM_VALUE = auto()     # 100-9999
    HIGH_VALUE = auto()       # 10000+
    KNOWN_CONSTANT = auto()   # Known mathematical constants


@dataclass
class ScoringConfig:
    """Configuration for the scoring engine."""
    # Trivial value filtering
    trivial_threshold: int = 10
    trivial_score: float = 0.0

    # Base scoring weights
    convergence_weight: float = 50.0
    non_trivial_bonus: float = 20.0

    # Value size bonuses
    large_value_threshold: int = 1000
    large_value_bonus: float = 10.0
    very_large_threshold: int = 100000
    very_large_bonus: float = 10.0

    # Pattern bonuses
    novelty_bonus: float = 15.0
    cycle_bonus: float = 12.0
    known_constant_bonus: float = 5.0

    # Property bonuses (from INTL-04 to INTL-07)
    palindrome_bonus: float = 10.0
    prime_bonus: float = 15.0
    perfect_power_bonus: float = 8.0
    repdigit_bonus: float = 5.0

    # Pipeline penalties
    length_penalty_per_op: float = 1.5

    # Cycle filtering
    min_cycle_length_for_trivial: int = 3  # Cycles < 3 with trivial values are filtered

    # Known mathematical constants
    known_constants: Set[int] = field(default_factory=lambda: {
        495, 6174, 1089, 10890, 99099, 26244, 109890, 1098900,
        153, 370, 371, 407,  # Armstrong numbers
        4, 16, 37, 58, 89, 145, 42, 20,  # Happy number cycle
    })


class ScoringEngineV2:
    """
    Revised scoring engine for SYNTRIAD v6.0.

    Implements intelligent scoring that:
    - Filters trivial attractors (0-9)
    - Rewards mathematically interesting properties
    - Uses configurable weights for fine-tuning
    - Integrates with ML feature importance

    Example:
        >>> engine = ScoringEngineV2()
        >>> score = engine.score(
        ...     attractor=6174,
        ...     convergence=0.98,
        ...     cycle_values=None,
        ...     pipeline_length=2,
        ...     properties={'is_palindrome': False}
        ... )
    """

    TRIVIAL_VALUES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def __init__(self, config: ScoringConfig = None):
        """
        Initialize the scoring engine.

        Args:
            config: ScoringConfig with scoring parameters
        """
        self.config = config or ScoringConfig()
        self._ml_feature_weights: Dict[str, float] = {}

    def categorize_attractor(self, value: int) -> AttractorCategory:
        """
        Categorize an attractor value.

        Args:
            value: The attractor value

        Returns:
            AttractorCategory classification
        """
        if value in self.config.known_constants:
            return AttractorCategory.KNOWN_CONSTANT
        if value < self.config.trivial_threshold:
            return AttractorCategory.TRIVIAL
        if value < 100:
            return AttractorCategory.LOW_VALUE
        if value < 10000:
            return AttractorCategory.MEDIUM_VALUE
        return AttractorCategory.HIGH_VALUE

    def is_trivial(self, value: int) -> bool:
        """Check if a value is trivial (0-9)."""
        return value in self.TRIVIAL_VALUES

    def is_trivial_cycle(self, cycle_values: List[int]) -> bool:
        """
        Check if a cycle contains only trivial values.

        A cycle is considered trivial if:
        - Length is 1-2 AND all values are trivial (0-9)

        Args:
            cycle_values: List of values in the cycle

        Returns:
            True if the cycle is trivial
        """
        if not cycle_values:
            return True

        # Short cycles with only trivial values are filtered
        if len(cycle_values) < self.config.min_cycle_length_for_trivial:
            return all(v in self.TRIVIAL_VALUES for v in cycle_values)

        return False

    def score(
        self,
        attractor: int,
        convergence: float,
        cycle_values: Optional[List[int]] = None,
        pipeline_length: int = 1,
        properties: Optional[Dict[str, bool]] = None,
        is_novel: bool = False,
        is_cycle: bool = False,
        endpoints: Optional[Dict[int, int]] = None,
        total_numbers: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the score for an attractor/discovery.

        Args:
            attractor: The main attractor value
            convergence: Convergence rate (0.0 to 1.0)
            cycle_values: Values in the cycle (if applicable)
            pipeline_length: Number of operations in pipeline
            properties: Dictionary of attractor properties
            is_novel: Whether this is a novel discovery
            is_cycle: Whether this converges to a cycle
            endpoints: Counter of endpoint values
            total_numbers: Total numbers tested

        Returns:
            Tuple of (score, breakdown_dict)
        """
        properties = properties or {}
        breakdown = {
            'base_score': 0.0,
            'convergence_score': 0.0,
            'value_bonuses': 0.0,
            'property_bonuses': 0.0,
            'novelty_bonus': 0.0,
            'cycle_bonus': 0.0,
            'penalties': 0.0,
            'filtered': False,
            'filter_reason': None
        }

        # Check for trivial attractor
        if self.is_trivial(attractor):
            breakdown['filtered'] = True
            breakdown['filter_reason'] = 'trivial_attractor'
            return self.config.trivial_score, breakdown

        # Check for trivial cycle
        if cycle_values and self.is_trivial_cycle(cycle_values):
            breakdown['filtered'] = True
            breakdown['filter_reason'] = 'trivial_cycle'
            return self.config.trivial_score, breakdown

        # Base convergence score
        breakdown['convergence_score'] = convergence * self.config.convergence_weight

        # Non-trivial bonus
        breakdown['value_bonuses'] += self.config.non_trivial_bonus

        # Value size bonuses
        if attractor >= self.config.large_value_threshold:
            breakdown['value_bonuses'] += self.config.large_value_bonus
        if attractor >= self.config.very_large_threshold:
            breakdown['value_bonuses'] += self.config.very_large_bonus

        # Known constant bonus
        if attractor in self.config.known_constants:
            breakdown['value_bonuses'] += self.config.known_constant_bonus

        # Novelty bonus
        if is_novel:
            breakdown['novelty_bonus'] = self.config.novelty_bonus

        # Cycle bonus
        if is_cycle:
            breakdown['cycle_bonus'] = self.config.cycle_bonus

        # Property bonuses
        if properties.get('is_palindrome'):
            breakdown['property_bonuses'] += self.config.palindrome_bonus
        if properties.get('is_prime'):
            breakdown['property_bonuses'] += self.config.prime_bonus
        if properties.get('is_perfect_power'):
            breakdown['property_bonuses'] += self.config.perfect_power_bonus
        if properties.get('is_repdigit'):
            breakdown['property_bonuses'] += self.config.repdigit_bonus

        # Pipeline length penalty
        breakdown['penalties'] = pipeline_length * self.config.length_penalty_per_op

        # Calculate total score
        total_score = (
            breakdown['convergence_score'] +
            breakdown['value_bonuses'] +
            breakdown['property_bonuses'] +
            breakdown['novelty_bonus'] +
            breakdown['cycle_bonus'] -
            breakdown['penalties']
        )

        return max(0.0, total_score), breakdown

    def score_from_result(
        self,
        endpoints: Dict[int, int],
        total_numbers: int,
        pipeline: List[int],
        known_attractors: Set[int],
        cycle_detected: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate score from raw evaluation results.

        Args:
            endpoints: Counter of endpoint values
            total_numbers: Total numbers tested
            pipeline: Pipeline operation codes
            known_attractors: Set of previously known attractors
            cycle_detected: Whether a cycle was detected

        Returns:
            Tuple of (score, breakdown_dict)
        """
        from attractor_analyzer import AttractorAnalyzer

        if not endpoints:
            return 0.0, {'filtered': True, 'filter_reason': 'no_endpoints'}

        # Get top attractor
        top_items = sorted(endpoints.items(), key=lambda x: -x[1])
        top_val, top_count = top_items[0]
        convergence = top_count / total_numbers if total_numbers > 0 else 0

        # Analyze properties
        analyzer = AttractorAnalyzer()
        properties = analyzer.get_properties(top_val)

        # Check novelty
        is_novel = top_val not in known_attractors and not self.is_trivial(top_val)

        return self.score(
            attractor=top_val,
            convergence=convergence,
            pipeline_length=len(pipeline),
            properties=properties,
            is_novel=is_novel,
            is_cycle=cycle_detected,
            endpoints=endpoints,
            total_numbers=total_numbers
        )

    def set_ml_feature_weights(self, weights: Dict[str, float]):
        """
        Set ML-derived feature importance weights.

        These weights can be used to adjust scoring based on
        learned feature importance from successful discoveries.

        Args:
            weights: Dictionary mapping feature names to importance weights
        """
        self._ml_feature_weights = weights.copy()

    def get_ml_adjusted_score(
        self,
        base_score: float,
        pipeline_features: Dict[str, Any]
    ) -> float:
        """
        Adjust score based on ML feature importance.

        Args:
            base_score: Original score
            pipeline_features: Features extracted from pipeline

        Returns:
            Adjusted score
        """
        if not self._ml_feature_weights:
            return base_score

        # Calculate feature-based adjustment
        adjustment = 0.0
        for feature, value in pipeline_features.items():
            if feature in self._ml_feature_weights:
                weight = self._ml_feature_weights[feature]
                if isinstance(value, bool):
                    adjustment += weight * 10 if value else 0
                elif isinstance(value, (int, float)):
                    adjustment += weight * value

        # Apply adjustment (max 20% bonus)
        max_adjustment = base_score * 0.2
        return base_score + min(adjustment, max_adjustment)

    def filter_trivial_results(
        self,
        discoveries: List[Dict[str, Any]],
        top_n: int = 100
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Filter and sort discoveries, removing trivial results.

        Args:
            discoveries: List of discovery dictionaries
            top_n: Number of top results to return

        Returns:
            Tuple of (filtered_discoveries, quality_metrics)
        """
        filtered = []
        trivial_count = 0
        total_count = len(discoveries)

        for disc in discoveries:
            attractor = disc.get('attractor', 0)
            cycle_values = disc.get('cycle_values')

            # Filter trivial attractors
            if self.is_trivial(attractor):
                trivial_count += 1
                continue

            # Filter trivial cycles
            if cycle_values and self.is_trivial_cycle(cycle_values):
                trivial_count += 1
                continue

            filtered.append(disc)

        # Sort by score
        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Calculate quality metrics
        top_results = filtered[:top_n]
        top_trivial = sum(1 for d in top_results if self.is_trivial(d.get('attractor', 0)))

        metrics = {
            'total_discoveries': total_count,
            'filtered_count': len(filtered),
            'trivial_filtered': trivial_count,
            'trivial_rate': trivial_count / total_count if total_count > 0 else 0,
            'top_n_trivial_rate': top_trivial / len(top_results) if top_results else 0,
            'quality_target_met': (top_trivial / len(top_results) if top_results else 1) < 0.05
        }

        return filtered[:top_n], metrics


def validate_scoring_requirements() -> Dict[str, bool]:
    """
    Validate that scoring requirements are met.

    Returns:
        Dictionary of requirement -> passed status
    """
    engine = ScoringEngineV2()
    results = {}

    # INTL-01: Trivial attractors filtered
    for trivial in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        score, breakdown = engine.score(trivial, convergence=1.0)
        if score != 0.0 or not breakdown['filtered']:
            results['INTL-01'] = False
            break
    else:
        results['INTL-01'] = True

    # INTL-02: Attractors < 10 get 0 score
    results['INTL-02'] = all(
        engine.score(v, convergence=1.0)[0] == 0.0
        for v in range(10)
    )

    # INTL-03: Trivial cycles filtered
    trivial_cycle = [1, 2]
    non_trivial_cycle = [1, 100]

    _, breakdown1 = engine.score(1, convergence=1.0, cycle_values=trivial_cycle)
    _, breakdown2 = engine.score(100, convergence=1.0, cycle_values=non_trivial_cycle)

    results['INTL-03'] = breakdown1['filtered'] and not breakdown2['filtered']

    # INTL-04 to INTL-07: Property bonuses
    score_base, _ = engine.score(12321, convergence=1.0)  # palindrome
    score_with_palindrome, _ = engine.score(
        12321, convergence=1.0, properties={'is_palindrome': True}
    )
    results['INTL-04'] = score_with_palindrome > score_base

    score_base, _ = engine.score(997, convergence=1.0)  # prime
    score_with_prime, _ = engine.score(
        997, convergence=1.0, properties={'is_prime': True}
    )
    results['INTL-05'] = score_with_prime > score_base

    score_base, _ = engine.score(256, convergence=1.0)  # perfect power
    score_with_power, _ = engine.score(
        256, convergence=1.0, properties={'is_perfect_power': True}
    )
    results['INTL-06'] = score_with_power > score_base

    score_base, _ = engine.score(1111, convergence=1.0)  # repdigit
    score_with_repdigit, _ = engine.score(
        1111, convergence=1.0, properties={'is_repdigit': True}
    )
    results['INTL-07'] = score_with_repdigit > score_base

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SYNTRIAD v6.0 - Scoring Engine v2 Validation")
    print("=" * 60)

    results = validate_scoring_requirements()

    print("\nRequirement Validation:")
    for req, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {req}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    # Demo scoring
    print("\n" + "=" * 60)
    print("Demo Scoring Examples")
    print("=" * 60)

    engine = ScoringEngineV2()

    examples = [
        (6174, 0.98, {'is_palindrome': False}),  # Kaprekar constant
        (1089, 0.95, {'is_palindrome': False}),  # 1089 trick
        (0, 1.0, {}),  # Trivial
        (12321, 0.90, {'is_palindrome': True}),  # Palindrome
        (997, 0.85, {'is_prime': True}),  # Prime
    ]

    for attractor, convergence, props in examples:
        score, breakdown = engine.score(
            attractor, convergence,
            pipeline_length=2,
            properties=props
        )
        status = "FILTERED" if breakdown['filtered'] else f"Score: {score:.1f}"
        print(f"  {attractor}: {status}")
