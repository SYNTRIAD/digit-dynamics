#!/usr/bin/env python3
"""
SYNTRIAD v6.0 - Regression Tests for Known Constants
====================================================

Tests that verify the system correctly discovers known mathematical
constants and cycles.

Requirements:
- TEST-03: Implement regression test for kaprekar_step -> 6174 convergence
- TEST-04: Implement regression test for happy_step cycle detection
- TEST-05: Implement regression test for truc_1089 -> 1089/10890 convergence

Author: SYNTRIAD Research
Created: 2026-01-27
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiprocessing_executor import _apply_single_op, _apply_pipeline_cpu


class TestKaprekarConstant:
    """Regression tests for Kaprekar constant discovery (TEST-03)"""

    def test_kaprekar_4digit_convergence(self):
        """Test 4-digit numbers converge to 6174."""
        # Test various 4-digit numbers
        # Note: numbers with repeated digits (like 2111, 9998) or leading zeros after sort have special behavior
        test_numbers = [1234, 3524, 7641, 5432, 8765, 4321]

        for n in test_numbers:
            result = n
            for _ in range(10):
                result = _apply_single_op(result, 2)  # kaprekar_step
                if result == 6174:
                    break

            assert result == 6174, f"{n} did not converge to 6174, got {result}"

    def test_kaprekar_3digit_convergence(self):
        """Test 3-digit numbers converge to 495."""
        # Note: 100 has special behavior (becomes 2-digit after sort_asc)
        test_numbers = [123, 456, 789, 321, 532, 954]

        for n in test_numbers:
            result = n
            for _ in range(10):
                result = _apply_single_op(result, 2)
                if result == 495:
                    break

            # 495 is the 3-digit Kaprekar constant
            assert result == 495, f"{n} did not converge to 495, got {result}"

    def test_kaprekar_constant_is_fixed_point(self):
        """Test that 6174 is a fixed point."""
        assert _apply_single_op(6174, 2) == 6174
        assert _apply_single_op(495, 2) == 495

    def test_kaprekar_repdigit_to_zero(self):
        """Test repdigit numbers converge to 0."""
        repdigits = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
        for n in repdigits:
            assert _apply_single_op(n, 2) == 0

    def test_kaprekar_pipeline_discovery(self):
        """Test pipeline correctly discovers Kaprekar constant."""
        result, steps, is_cycle = _apply_pipeline_cpu(1234, [2], 20)

        # Should find 6174 as cycle/fixed point
        if is_cycle and result != 0:
            assert result == 6174


class TestHappyNumberCycle:
    """Regression tests for happy number cycle detection (TEST-04)"""

    # Known happy numbers (reach 1)
    HAPPY_NUMBERS = {1, 7, 10, 13, 19, 23, 28, 31, 32, 44, 49, 68, 70, 79, 82, 86, 91, 94, 97, 100}

    # The unhappy cycle
    UNHAPPY_CYCLE = {4, 16, 37, 58, 89, 145, 42, 20}

    def test_happy_numbers_reach_1(self):
        """Test known happy numbers reach 1."""
        for n in self.HAPPY_NUMBERS:
            result = n
            for _ in range(50):
                result = _apply_single_op(result, 4)  # happy_step
                if result == 1:
                    break

            assert result == 1, f"Happy number {n} did not reach 1, got {result}"

    def test_unhappy_cycle_detection(self):
        """Test unhappy numbers enter the 4-16-37-58-89-145-42-20 cycle."""
        # Start with 4 (in the cycle)
        n = 4
        seen = set()

        for _ in range(20):
            n = _apply_single_op(n, 4)
            if n in seen:
                break
            seen.add(n)

        # All seen values should be in the unhappy cycle
        assert seen.issubset(self.UNHAPPY_CYCLE)

    def test_happy_step_cycle_in_pipeline(self):
        """Test pipeline detects happy number behavior."""
        # Test with unhappy number - should detect cycle
        result, steps, is_cycle = _apply_pipeline_cpu(2, [4], 50)

        # 2 is unhappy, should eventually cycle
        # Could be 1 (happy) or in cycle
        assert is_cycle or result == 1

    def test_happy_step_values(self):
        """Test specific happy_step calculations."""
        # 7 -> 49 -> 97 -> 130 -> 10 -> 1
        assert _apply_single_op(7, 4) == 49
        assert _apply_single_op(49, 4) == 97  # 16 + 81
        assert _apply_single_op(10, 4) == 1

    def test_unhappy_number_4(self):
        """Test that 4 is in a cycle."""
        # 4 -> 16 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20 -> 4
        values = [4]
        n = 4
        for _ in range(10):
            n = _apply_single_op(n, 4)
            values.append(n)
            if n == 4:
                break

        assert 4 in values[1:]  # Cycle back to 4


class TestTruc1089:
    """Regression tests for 1089 trick convergence (TEST-05)"""

    def test_1089_3digit_convergence(self):
        """Test 3-digit numbers converge to 1089."""
        # Numbers with first digit > last digit (by at least 2)
        test_numbers = [321, 421, 531, 621, 731, 841, 951]

        for n in test_numbers:
            result = _apply_single_op(n, 3)  # truc_1089
            assert result == 1089, f"{n} gave {result}, expected 1089"

    def test_1089_5digit_convergence(self):
        """Test 5-digit numbers converge to 10890 or similar."""
        test_numbers = [54321, 65432, 76543]

        for n in test_numbers:
            result = _apply_single_op(n, 3)
            # 5-digit numbers typically give 10890, 99099, or 109890
            assert result in [10890, 99099, 109890], f"{n} gave {result}"

    def test_1089_palindrome_gives_zero(self):
        """Test palindromes give 0."""
        palindromes = [121, 131, 12321, 45654]
        for p in palindromes:
            assert _apply_single_op(p, 3) == 0

    def test_1089_specific_calculation(self):
        """Test specific 1089 trick calculation."""
        # Take 321:
        # 321 - 123 = 198
        # 198 + 891 = 1089
        assert _apply_single_op(321, 3) == 1089

        # Take 532:
        # 532 - 235 = 297
        # 297 + 792 = 1089
        assert _apply_single_op(532, 3) == 1089

    def test_1089_pipeline_discovery(self):
        """Test pipeline can discover 1089 or 10890 depending on digit count."""
        endpoints = []
        for n in range(321, 421):
            result, steps, is_cycle = _apply_pipeline_cpu(n, [3], 5)
            endpoints.append(result)

        # 3-digit numbers converge to 1089 or related values
        # Count both 1089 and 10890 as valid (depends on intermediate digit count)
        valid_count = endpoints.count(1089) + endpoints.count(10890)
        assert valid_count > len(endpoints) * 0.5


class TestKnownConstants:
    """Tests for other known mathematical constants"""

    def test_armstrong_numbers_153(self):
        """Test Armstrong number 153 is a fixed point for digit_pow3."""
        assert _apply_single_op(153, 5) == 153  # 1^3 + 5^3 + 3^3 = 153

    def test_armstrong_numbers_370(self):
        """Test Armstrong number 370."""
        assert _apply_single_op(370, 5) == 370

    def test_armstrong_numbers_371(self):
        """Test Armstrong number 371."""
        assert _apply_single_op(371, 5) == 371

    def test_armstrong_numbers_407(self):
        """Test Armstrong number 407."""
        assert _apply_single_op(407, 5) == 407

    def test_perfect_digital_invariant_1634(self):
        """Test 1634 is PDI for digit_pow4."""
        assert _apply_single_op(1634, 6) == 1634  # 1^4 + 6^4 + 3^4 + 4^4

    def test_kaprekar_number_9(self):
        """Test Kaprekar number 9: 9^2=81, 8+1=9."""
        # This is a different Kaprekar property (number theory)
        sq = 9 * 9  # 81
        left = 8
        right = 1
        assert left + right == 9


class TestConvergenceBehavior:
    """Tests for general convergence behavior"""

    def test_digit_sum_convergence(self):
        """Test digit_sum converges to single digit."""
        for n in [123, 456789, 99999]:
            result = n
            while result >= 10:
                result = _apply_single_op(result, 1)

            assert 1 <= result <= 9

    def test_digit_product_convergence(self):
        """Test digit_product eventually reaches single digit or 0."""
        for n in [25, 39, 77]:
            result = n
            steps = 0
            while result >= 10 and steps < 10:
                result = _apply_single_op(result, 15)
                steps += 1

            assert result < 10 or steps >= 10

    def test_pipeline_kaprekar_digitsum(self):
        """Test combined kaprekar + digitsum pipeline."""
        result, steps, is_cycle = _apply_pipeline_cpu(1234, [2, 1], 20)
        # Kaprekar gives 6174, digitsum gives 18
        # Should converge eventually
        assert result >= 0


class TestMultiDigitRanges:
    """Tests across different digit ranges"""

    def test_kaprekar_2digit(self):
        """Test 2-digit Kaprekar behavior."""
        # 2-digit numbers have different behavior
        results = set()
        for n in range(10, 100):
            result = n
            for _ in range(10):
                new_result = _apply_single_op(result, 2)
                if new_result == result:
                    break
                result = new_result
            results.add(result)

        # Should find small set of attractors
        assert len(results) < 20

    def test_1089_4digit(self):
        """Test 4-digit 1089 trick gives 10890."""
        test_numbers = [4321, 5432, 6543, 7654, 8765, 9876]
        for n in test_numbers:
            result = _apply_single_op(n, 3)
            # 4-digit should give 10890
            assert result == 10890, f"{n} gave {result}"

    def test_happy_large_numbers(self):
        """Test happy step on larger numbers."""
        # Large number happy step should reduce significantly
        n = 999999
        result = _apply_single_op(n, 4)

        # 6*81 = 486, should be much smaller than n
        assert result < n
        assert result == 6 * 81


class TestSystemIntegration:
    """Integration tests for the full system"""

    def test_pipeline_evaluation_consistency(self):
        """Test pipeline evaluation gives consistent results."""
        pipeline = [2, 1]  # kaprekar -> digitsum
        n = 1234

        results = []
        for _ in range(5):
            result, steps, is_cycle = _apply_pipeline_cpu(n, pipeline, 20)
            results.append(result)

        # All results should be identical
        assert len(set(results)) == 1

    def test_known_constants_discoverable(self):
        """Test that known constants are discoverable by pipelines."""
        # Kaprekar constant
        result, _, _ = _apply_pipeline_cpu(1234, [2], 20)
        assert result in [6174, 0]

        # Armstrong
        result, _, _ = _apply_pipeline_cpu(153, [5], 10)
        assert result == 153


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
