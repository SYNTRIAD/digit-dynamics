"""
Tests for Engine vNext M2: Feature Extractor + Conjecture Mining

Test categories:
  1. NumberFeatures: digit decomposition, profiles, edge cases
  2. OrbitAnalyzer: orbit tracing, basin analysis
  3. ConjectureMiner: counting, invariant, universality, structure templates
  4. Falsifier: falsification and refinement
  5. Integration: full pipeline from features to conjectures
"""

import pytest

from pipeline_dsl import (
    OperationRegistry, Pipeline, DomainPolicy, PipelineRunner,
)
from feature_extractor import (
    NumberFeatures, NumberProfile, OrbitAnalyzer, OrbitProfile,
    ConjectureMiner, ConjectureType, Conjecture,
    Falsifier, FalsificationResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reg():
    return OperationRegistry()

@pytest.fixture
def nf():
    return NumberFeatures(base=10)

@pytest.fixture
def oa(reg):
    return OrbitAnalyzer(reg)

@pytest.fixture
def miner(reg):
    return ConjectureMiner(reg)

@pytest.fixture
def falsifier(reg):
    return Falsifier(reg)


# =============================================================================
# 1. NUMBER FEATURES
# =============================================================================

@pytest.mark.unit
class TestNumberFeatures:

    def test_digits_basic(self, nf):
        assert nf.digits(6174) == [6, 1, 7, 4]
        assert nf.digits(0) == [0]
        assert nf.digits(100) == [1, 0, 0]

    def test_digit_sum(self, nf):
        p = nf.compute(6174)
        assert p.digit_sum == 18

    def test_digit_product(self, nf):
        p = nf.compute(234)
        assert p.digit_product == 24

    def test_digital_root(self, nf):
        p = nf.compute(6174)
        assert p.digital_root == 9  # 6+1+7+4=18 → 1+8=9

    def test_mod_invariants(self, nf):
        p = nf.compute(6174)
        assert p.mod_b_minus_1 == 0  # 6174 mod 9 = 0
        assert p.mod_b_plus_1 == 3   # 6174 mod 11 = 3

    def test_palindrome(self, nf):
        assert nf.compute(12321).is_palindrome is True
        assert nf.compute(1234).is_palindrome is False

    def test_repdigit(self, nf):
        assert nf.compute(1111).is_repdigit is True
        assert nf.compute(1112).is_repdigit is False

    def test_sortedness_ascending(self, nf):
        p = nf.compute(1234)
        assert p.sortedness_tau == 1.0

    def test_sortedness_descending(self, nf):
        p = nf.compute(4321)
        assert p.sortedness_tau == -1.0

    def test_sortedness_mixed(self, nf):
        p = nf.compute(6174)
        assert -1.0 <= p.sortedness_tau <= 1.0

    def test_complement_distance(self, nf):
        p = nf.compute(1234)
        # complement_9(1234) = 8765, distance = |1234 - 8765| = 7531
        assert p.complement_distance == 7531

    def test_digit_entropy_repdigit(self, nf):
        p = nf.compute(1111)
        assert p.digit_entropy == 0.0  # all same digits

    def test_digit_entropy_varied(self, nf):
        p = nf.compute(1234)
        assert p.digit_entropy > 0  # all different digits → max entropy for 4

    def test_digit_range(self, nf):
        p = nf.compute(1928)
        assert p.digit_range == 8  # 9 - 1

    def test_has_zero_digit(self, nf):
        assert nf.compute(1024).has_zero_digit is True
        assert nf.compute(1234).has_zero_digit is False

    def test_zero(self, nf):
        p = nf.compute(0)
        assert p.digit_sum == 0
        assert p.num_digits == 1

    def test_padded_digits(self, nf):
        p = nf.compute(42, num_digits=4)
        assert p.num_digits == 4

    def test_to_dict(self, nf):
        d = nf.compute(6174).to_dict()
        assert "n" in d
        assert "digit_entropy" in d
        assert d["n"] == 6174

    def test_base_16(self):
        nf16 = NumberFeatures(base=16)
        assert nf16.digits(255) == [15, 15]  # 0xFF
        p = nf16.compute(255)
        assert p.is_repdigit is True


# =============================================================================
# 2. ORBIT ANALYZER
# =============================================================================

@pytest.mark.integration
class TestOrbitAnalyzer:

    def test_kaprekar_fp_orbit(self, oa, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        orbit = oa.trace_orbit(pipe, 6174)
        assert orbit.attractor == 6174
        assert orbit.is_fixed_point is True
        assert orbit.transient_length == 0
        assert orbit.cycle_length == 1

    def test_kaprekar_convergent_orbit(self, oa, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        orbit = oa.trace_orbit(pipe, 3087)
        assert orbit.attractor == 6174
        assert orbit.transient_length > 0
        assert orbit.is_fixed_point is True

    def test_orbit_has_contraction_ratios(self, oa, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        orbit = oa.trace_orbit(pipe, 1234)
        assert len(orbit.contraction_ratios) > 0

    def test_orbit_digit_sum_trajectory(self, oa, reg):
        pipe = Pipeline.parse("digit_sum", registry=reg)
        orbit = oa.trace_orbit(pipe, 99)
        assert orbit.digit_sum_trajectory[0] == 18  # ds(99) = 18
        assert all(isinstance(v, int) for v in orbit.digit_sum_trajectory)

    def test_orbit_mod9_trajectory(self, oa, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        orbit = oa.trace_orbit(pipe, 4321)
        assert all(0 <= v < 9 for v in orbit.mod9_trajectory)

    def test_orbit_to_dict(self, oa, reg):
        pipe = Pipeline.parse("reverse", registry=reg)
        orbit = oa.trace_orbit(pipe, 1234)
        d = orbit.to_dict()
        assert "start" in d
        assert "attractor" in d
        assert "avg_contraction" in d

    def test_basin_analysis_small(self, oa, reg):
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        domain = DomainPolicy(base=10, digit_length=3)
        basin = oa.analyze_basin(pipe, domain)
        assert basin["num_values"] > 0
        assert basin["num_attractors"] > 0
        assert "avg_transient" in basin


# =============================================================================
# 3. CONJECTURE MINER
# =============================================================================

@pytest.mark.exhaustive
class TestConjectureMiner:

    def test_mine_counting_kaprekar(self, miner):
        conjs = miner.mine_counting("kaprekar_step", [3, 4, 5])
        # Should find something (FP count varies, so likely monotonicity)
        assert isinstance(conjs, list)

    def test_mine_invariants_kaprekar(self, miner):
        conjs = miner.mine_invariants("kaprekar_step", [3, 4, 5])
        # Kaprekar FPs are divisible by 9
        mod9_conj = [c for c in conjs if c.parameters.get("modulus") == 9]
        assert len(mod9_conj) > 0
        assert mod9_conj[0].ctype == ConjectureType.MODULAR

    def test_mine_invariants_1089(self, miner):
        conjs = miner.mine_invariants("truc_1089", [3, 4, 5])
        # 1089 FPs are divisible by 9 and 11
        mod9 = [c for c in conjs if c.parameters.get("modulus") == 9]
        mod11 = [c for c in conjs if c.parameters.get("modulus") == 11]
        assert len(mod9) > 0
        assert len(mod11) > 0

    def test_mine_universality(self, miner):
        conjs = miner.mine_universality("kaprekar_step", [3, 4], exclude_repdigits=True)
        assert isinstance(conjs, list)

    def test_mine_structure(self, miner):
        conjs = miner.mine_structure("truc_1089", [3, 4, 5])
        assert isinstance(conjs, list)

    def test_mine_all_returns_ranked(self, miner):
        conjs = miner.mine_all("kaprekar_step", [3, 4, 5])
        assert isinstance(conjs, list)
        if len(conjs) >= 2:
            assert conjs[0].score >= conjs[1].score

    def test_conjecture_has_required_fields(self, miner):
        conjs = miner.mine_invariants("kaprekar_step", [3, 4])
        if conjs:
            c = conjs[0]
            assert c.ctype is not None
            assert c.statement
            assert c.quantifier
            assert c.predicate
            assert c.pipeline == "kaprekar_step"
            assert 0.0 <= c.confidence <= 1.0

    def test_conjecture_to_dict(self, miner):
        conjs = miner.mine_invariants("kaprekar_step", [3, 4])
        if conjs:
            d = conjs[0].to_dict()
            assert "type" in d
            assert "statement" in d
            assert "score" in d

    def test_conjecture_str(self, miner):
        conjs = miner.mine_invariants("kaprekar_step", [3, 4])
        if conjs:
            s = str(conjs[0])
            assert "modular" in s or "invariant" in s


# =============================================================================
# 4. FALSIFIER
# =============================================================================

@pytest.mark.exhaustive
class TestFalsifier:

    def test_surviving_conjecture(self, miner, falsifier):
        """Kaprekar mod-9 should survive extension to k=6."""
        conjs = miner.mine_invariants("kaprekar_step", [3, 4, 5])
        mod9 = [c for c in conjs if c.parameters.get("modulus") == 9]
        assert len(mod9) > 0
        fr = falsifier.test_modular(mod9[0], [6])
        assert fr.falsified is False

    def test_1089_mod9_survives(self, miner, falsifier):
        conjs = miner.mine_invariants("truc_1089", [3, 4, 5])
        mod9 = [c for c in conjs if c.parameters.get("modulus") == 9]
        assert len(mod9) > 0
        fr = falsifier.test_modular(mod9[0], [6, 7])
        assert fr.falsified is False

    def test_1089_mod11_survives(self, miner, falsifier):
        conjs = miner.mine_invariants("truc_1089", [3, 4, 5])
        mod11 = [c for c in conjs if c.parameters.get("modulus") == 11]
        assert len(mod11) > 0
        fr = falsifier.test_modular(mod11[0], [6])
        assert fr.falsified is False

    def test_falsification_result_str(self, miner, falsifier):
        conjs = miner.mine_invariants("kaprekar_step", [3, 4])
        if conjs:
            fr = falsifier.test_modular(conjs[0], [5])
            s = str(fr)
            assert "SURVIVED" in s or "FALSIFIED" in s


# =============================================================================
# 5. INTEGRATION
# =============================================================================

@pytest.mark.exhaustive
class TestIntegration:

    def test_full_pipeline_features_to_conjectures(self, reg, miner):
        """End-to-end: features + mining + ranking for kaprekar."""
        # Number features
        nf = NumberFeatures(base=10)
        p = nf.compute(6174)
        assert p.mod_b_minus_1 == 0  # 6174 ≡ 0 mod 9

        # Orbit
        oa = OrbitAnalyzer(reg)
        pipe = Pipeline.parse("kaprekar_step", registry=reg)
        orbit = oa.trace_orbit(pipe, 3087)
        assert orbit.attractor == 6174

        # Mine
        conjs = miner.mine_all("kaprekar_step", [3, 4, 5])
        assert len(conjs) > 0
        # Top conjecture should be about mod 9
        top = conjs[0]
        assert top.score > 0

    def test_full_pipeline_1089(self, reg, miner, falsifier):
        """End-to-end for truc_1089 with falsification."""
        conjs = miner.mine_all("truc_1089", [3, 4, 5], exclude_repdigits=False)
        assert len(conjs) > 0

        # Falsify against k=6
        for c in conjs[:3]:
            if c.ctype == ConjectureType.MODULAR:
                fr = falsifier.test_modular(c, [6])
                assert isinstance(fr, FalsificationResult)

    def test_conjecture_json_serializable(self, miner):
        """All conjectures must be JSON-serializable."""
        conjs = miner.mine_all("kaprekar_step", [3, 4, 5])
        import json
        for c in conjs:
            s = json.dumps(c.to_dict(), sort_keys=True)
            assert isinstance(s, str)
            d = json.loads(s)
            assert d["type"] in ["counting", "invariant", "universality",
                                  "structure", "monotonicity", "modular"]
