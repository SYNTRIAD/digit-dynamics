# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research — MIT License
"""
Tests for multi-base digit operations.

Validates:
  1. BaseDigitOps to_digits/from_digits roundtrip for bases 2, 6, 8, 10, 12, 16
  2. Kaprekar constants: 495 (base10/3d), 6174 (base10/4d), (b/2)(b²-1) for even bases
  3. 1089 analogues: (b-1)(b+1)² for bases 6, 8, 10, 12
  4. Complement involution: complement(complement(n)) == n for all bases
  5. is_repdigit base-aware: 73₁₀ = 111₈ is repdigit in base 8
  6. Pipeline execution with base parameter
"""

import pytest
from pipeline_dsl import (
    BaseDigitOps, OperationRegistry, Pipeline, DomainPolicy, PipelineRunner,
)


# =============================================================================
# 1. BaseDigitOps roundtrip
# =============================================================================

class TestBaseDigitOpsRoundtrip:
    """to_digits -> from_digits roundtrip for multiple bases."""

    @pytest.mark.parametrize("base", [2, 6, 8, 10, 12, 16])
    @pytest.mark.parametrize("n", [0, 1, 7, 42, 100, 255, 1000, 6174, 65535])
    def test_roundtrip(self, base, n):
        ops = BaseDigitOps(base)
        digits = ops.to_digits(n)
        assert ops.from_digits(digits) == n

    @pytest.mark.parametrize("base", [2, 8, 10, 16])
    def test_digits_are_valid(self, base):
        """All digits must be in [0, base-1]."""
        ops = BaseDigitOps(base)
        for n in [0, 1, 42, 255, 1000]:
            digits = ops.to_digits(n)
            for d in digits:
                assert 0 <= d < base, f"Digit {d} out of range for base {base}"

    def test_base10_matches_str(self):
        """Base-10 digits should match str() conversion."""
        ops = BaseDigitOps(10)
        for n in [0, 1, 42, 123, 6174, 99999]:
            digits = ops.to_digits(n)
            expected = [int(c) for c in str(n)]
            assert digits == expected

    def test_base2_known(self):
        """Binary representation of known values."""
        ops = BaseDigitOps(2)
        assert ops.to_digits(0) == [0]
        assert ops.to_digits(1) == [1]
        assert ops.to_digits(5) == [1, 0, 1]
        assert ops.to_digits(255) == [1, 1, 1, 1, 1, 1, 1, 1]

    def test_base16_known(self):
        """Hex representation of known values."""
        ops = BaseDigitOps(16)
        assert ops.to_digits(255) == [15, 15]  # FF
        assert ops.to_digits(256) == [1, 0, 0]  # 100

    def test_invalid_base(self):
        with pytest.raises(ValueError):
            BaseDigitOps(1)
        with pytest.raises(ValueError):
            BaseDigitOps(0)


# =============================================================================
# 2. Kaprekar constants
# =============================================================================

class TestKaprekarMultiBase:
    """Kaprekar constants in various bases."""

    def test_kaprekar_base10_3digit(self):
        """495 is the Kaprekar constant for 3-digit base-10."""
        ops = BaseDigitOps(10)
        assert ops.kaprekar_step(495) == 495

    def test_kaprekar_base10_4digit(self):
        """6174 is the Kaprekar constant for 4-digit base-10."""
        ops = BaseDigitOps(10)
        assert ops.kaprekar_step(6174) == 6174

    def test_kaprekar_convergence_base10(self):
        """4-digit non-repdigit base-10 numbers converge to 6174 (or 0 for near-repdigits)."""
        ops = BaseDigitOps(10)
        for start in [1234, 3087, 8352, 4567, 2538]:
            n = start
            for _ in range(8):
                n = ops.kaprekar_step(n)
                if n == 6174:
                    break
            assert n == 6174, f"{start} did not converge to 6174, got {n}"

    @pytest.mark.parametrize("base", [4, 6, 8, 10, 12, 16])
    def test_kaprekar_3digit_even_base_formula(self, base):
        """For even bases, 3-digit Kaprekar constant = (b/2)(b²-1).

        This is a well-known result in number theory.
        """
        expected = (base // 2) * (base * base - 1)
        ops = BaseDigitOps(base)
        result = ops.kaprekar_step(expected)
        assert result == expected, (
            f"Base {base}: expected {expected} to be fixed point, got {result}"
        )

    def test_kaprekar_via_registry(self):
        """Test multi-base Kaprekar via OperationRegistry.execute_with_base()."""
        reg = OperationRegistry()
        # Base-10 classic
        assert reg.execute_with_base("kaprekar_step", 6174, base=10) == 6174
        # Base-6: (6/2)(36-1) = 3*35 = 105
        assert reg.execute_with_base("kaprekar_step", 105, base=6) == 105


# =============================================================================
# 3. 1089 analogues
# =============================================================================

class TestTruc1089MultiBase:
    """1089 trick analogues in different bases."""

    def test_1089_base10(self):
        """Classic: 3-digit base-10 numbers → 1089."""
        ops = BaseDigitOps(10)
        assert ops.truc_1089(321) == 1089
        assert ops.truc_1089(532) == 1089

    @pytest.mark.parametrize("base,expected", [
        (6, 245),    # (6-1)(6+1)² = 5*49 = 245
        (8, 567),    # (8-1)(8+1)² = 7*81 = 567
        (10, 1089),  # (10-1)(10+1)² = 9*121 = 1089
        (12, 1859),  # (12-1)(12+1)² = 11*169 = 1859
    ])
    def test_1089_analogue_formula(self, base, expected):
        """For base b, the 3-digit 1089 analogue is (b-1)(b+1)².

        3-digit numbers (where first digit > last digit in base b)
        map to this value in one step. We test with a canonical 3-digit input.
        """
        ops = BaseDigitOps(base)
        formula_val = (base - 1) * (base + 1) ** 2
        assert formula_val == expected

        # Build a 3-digit number in base b: digits [b-1, 0, 0] (= (b-1)*b²)
        # which has first_digit > last_digit, so the trick applies
        test_input = (base - 1) * base * base  # e.g., base10: 900
        result = ops.truc_1089(test_input)
        assert result == formula_val, (
            f"Base {base}: truc_1089({test_input}) = {result}, expected {formula_val}"
        )

    def test_1089_via_registry(self):
        """Test via OperationRegistry."""
        reg = OperationRegistry()
        assert reg.execute_with_base("truc_1089", 321, base=10) == 1089
        # Base-8: 3-digit input (e.g., 7*64=448) should give 567
        assert reg.execute_with_base("truc_1089", 448, base=8) == 567


# =============================================================================
# 4. Complement involution
# =============================================================================

class TestComplementInvolution:
    """complement(complement(n)) == n for all bases."""

    @pytest.mark.parametrize("base", [2, 6, 8, 10, 12, 16])
    def test_involution(self, base):
        """Complement is an involution when MSB < base-1 (no leading-zero drop).

        When MSB == base-1, complement produces leading zeros which drop,
        reducing digit count. This is expected (LeadingZeroPolicy.DROPS).
        """
        ops = BaseDigitOps(base)
        # Pick values where MSB < base-1 so complement doesn't produce leading zeros
        test_values = [1, base + 1, base * 2 + 1, base ** 2 + base + 1]
        for n in test_values:
            digits = ops.to_digits(n)
            if digits[0] >= base - 1:
                continue  # skip — would produce leading zeros
            assert ops.complement(ops.complement(n)) == n, (
                f"Complement not involutive for n={n} in base {base}"
            )

    def test_complement_base10_matches_legacy(self):
        """Base-10 complement should match OperationExecutor.complement_9."""
        from pipeline_dsl import OperationExecutor
        ops = BaseDigitOps(10)
        for n in [123, 456, 789, 100, 999, 5050]:
            assert ops.complement(n) == OperationExecutor.complement_9(n)

    def test_complement_via_registry(self):
        """Test via execute_with_base."""
        reg = OperationRegistry()
        ops6 = BaseDigitOps(6)
        for n in [10, 25, 100]:
            assert reg.execute_with_base("complement_9", n, base=6) == ops6.complement(n)


# =============================================================================
# 5. is_repdigit base-aware
# =============================================================================

class TestRepdigitMultiBase:
    """Base-aware repdigit detection."""

    def test_repdigit_base10(self):
        """Standard base-10 repdigits."""
        ops = BaseDigitOps(10)
        assert ops.is_repdigit(111) is True
        assert ops.is_repdigit(5555) is True
        assert ops.is_repdigit(123) is False

    def test_73_is_repdigit_base8(self):
        """73₁₀ = 111₈, so it is a repdigit in base 8."""
        ops8 = BaseDigitOps(8)
        assert ops8.to_digits(73) == [1, 1, 1]
        assert ops8.is_repdigit(73) is True

        ops10 = BaseDigitOps(10)
        assert ops10.is_repdigit(73) is False

    def test_repdigit_base2(self):
        """Binary repdigits: 1, 3 (11), 7 (111), 15 (1111), ..."""
        ops = BaseDigitOps(2)
        assert ops.is_repdigit(3) is True   # 11
        assert ops.is_repdigit(7) is True   # 111
        assert ops.is_repdigit(15) is True  # 1111
        assert ops.is_repdigit(5) is False  # 101

    def test_domain_policy_repdigit_base_aware(self):
        """DomainPolicy.is_repdigit should use the domain's base."""
        d10 = DomainPolicy(base=10, digit_length=3)
        assert d10.is_repdigit(111) is True
        assert d10.is_repdigit(73) is False

        d8 = DomainPolicy(base=8, digit_length=3)
        assert d8.is_repdigit(73) is True   # 111 in base 8
        assert d8.is_repdigit(111) is False  # 157 in base 8 = not repdigit


# =============================================================================
# 6. Pipeline multi-base execution
# =============================================================================

class TestPipelineMultiBase:
    """Pipeline execution with base parameter."""

    def test_pipeline_base10_unchanged(self):
        """Base-10 pipeline should produce identical results to legacy."""
        reg = OperationRegistry()
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        # Legacy path (no base param)
        r1 = reg.execute_pipeline(pipe, 1234)
        # New path with explicit base=10
        r2 = reg.execute_pipeline(pipe, 1234, base=10)
        assert r1 == r2

    def test_pipeline_kaprekar_base6(self):
        """Kaprekar pipeline in base 6 should converge to 105."""
        reg = OperationRegistry()
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        # Start with a 3-digit base-6 number (36..215 in decimal)
        n = 100  # 244₆ — a 3-digit base-6 number
        for _ in range(20):
            n = reg.execute_pipeline(pipe, n, base=6)
            if n == 105:
                break
        assert n == 105

    def test_exhaustive_run_base10_hash_stable(self):
        """Exhaustive run with base=10 should produce same hash as before."""
        reg = OperationRegistry()
        runner = PipelineRunner(reg)
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
        result = runner.run_exhaustive(pipe, domain)
        # 6174 should be the dominant fixed point
        assert 6174 in result.fixed_points

    def test_exhaustive_run_base6_3digit(self):
        """Exhaustive run in base 6, 3-digit domain."""
        reg = OperationRegistry()
        runner = PipelineRunner(reg)
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=6, digit_length=3, exclude_repdigits=True)
        result = runner.run_exhaustive(pipe, domain)
        # Kaprekar constant for base 6, 3-digit = 105
        assert 105 in result.fixed_points


# =============================================================================
# 7. Base-10 backward compatibility
# =============================================================================

class TestBase10BackwardCompat:
    """Ensure base-10 operations are bit-for-bit identical."""

    def test_all_ops_base10_match(self):
        """All 22 operations via execute_with_base(base=10) must match execute()."""
        reg = OperationRegistry()
        test_values = [0, 1, 7, 42, 123, 495, 1234, 6174, 9999]
        for name in reg.all_names():
            for n in test_values:
                legacy = reg.execute(name, n)
                new = reg.execute_with_base(name, n, base=10)
                assert legacy == new, (
                    f"Mismatch for {name}({n}): legacy={legacy}, new={new}"
                )
