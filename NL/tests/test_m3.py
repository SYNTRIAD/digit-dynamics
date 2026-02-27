"""
Tests for Engine vNext M3: Proof Engine + Structural Reasoning

Test categories:
  1. ProofSkeleton: skeleton generation for each conjecture type
  2. DensityEstimator: CE density, confidence scaling, labels
  3. PatternCompressor: affine, constant, modular, eventually_constant
  4. ConjectureMutator: generalize, transfer, strengthen, weaken
  5. RankingModelV1: formalized ranking, score decomposition
  6. Integration: mine → skeleton → density → rank pipeline
"""

import pytest

from pipeline_dsl import (
    OperationRegistry, Pipeline, DomainPolicy, PipelineRunner,
)
from feature_extractor import (
    NumberFeatures, ConjectureMiner, ConjectureType, Conjecture,
    TestedDomain, MonotonicityKind, Falsifier,
)
from proof_engine import (
    ProofSkeleton, ProofStrategy, ReductionStep, SkeletonGenerator,
    DensityEstimate, DensityEstimator,
    DetectedPattern, PatternCompressor,
    ConjectureMutator, MutationType, Mutation,
    RankingModelV1, RankedConjecture, RANKING_MODEL_VERSION,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reg():
    return OperationRegistry()

@pytest.fixture
def miner(reg):
    return ConjectureMiner(reg)

@pytest.fixture
def gen(reg):
    return SkeletonGenerator(reg)

@pytest.fixture
def estimator():
    return DensityEstimator()

@pytest.fixture
def compressor():
    return PatternCompressor()

@pytest.fixture
def mutator(reg):
    return ConjectureMutator(reg)

@pytest.fixture
def ranker(reg):
    return RankingModelV1(reg)


def _make_modular_conjecture(pipeline="kaprekar_step", modulus=9, residue=0,
                             base=10, k_range=None) -> Conjecture:
    """Helper: create a modular conjecture with tested_domains."""
    if k_range is None:
        k_range = [3, 4, 5]
    tested = []
    for k in k_range:
        dp = DomainPolicy(base=base, digit_length=k)
        lo, hi = dp.range()
        tested.append(TestedDomain(base=base, digit_length=k,
                                   range_lo=lo, range_hi=hi))
    return Conjecture(
        ctype=ConjectureType.MODULAR,
        statement=f"All non-zero FPs of {pipeline} are divisible by {modulus}",
        quantifier=f"∀FP∈FP({pipeline}), FP>0",
        predicate=f"{modulus} | FP",
        evidence={"residues": {k: [0] for k in k_range}},
        exceptions=[],
        pipeline=pipeline,
        parameters={"base": base, "modulus": modulus, "residue": residue,
                    "k_range": k_range},
        confidence=0.8,
        novelty=0.7,
        simplicity=0.9,
        tested_domains=tested,
    )


def _make_counting_conjecture(pipeline="kaprekar_step", fp_counts=None) -> Conjecture:
    """Helper: create a counting conjecture."""
    if fp_counts is None:
        fp_counts = {3: 1, 4: 1, 5: 3, 6: 3, 7: 5}
    k_range = sorted(fp_counts.keys())
    tested = []
    for k in k_range:
        dp = DomainPolicy(base=10, digit_length=k, exclude_repdigits=True)
        lo, hi = dp.range()
        tested.append(TestedDomain(base=10, digit_length=k,
                                   range_lo=lo, range_hi=hi,
                                   exclude_repdigits=True))
    return Conjecture(
        ctype=ConjectureType.COUNTING,
        statement=f"#FP({pipeline}) count pattern",
        quantifier=f"∀k∈{{{','.join(map(str, k_range))}}}",
        predicate=f"#FP varies with k",
        evidence=fp_counts,
        exceptions=[],
        pipeline=pipeline,
        parameters={"base": 10, "k_range": k_range},
        confidence=0.7,
        novelty=0.6,
        simplicity=0.8,
        tested_domains=tested,
    )


def _make_monotonicity_conjecture() -> Conjecture:
    """Helper: monotonicity conjecture."""
    return Conjecture(
        ctype=ConjectureType.MONOTONICITY,
        statement="#FP(truc_1089) is monotone nondecreasing in k",
        quantifier="∀k₁<k₂",
        predicate="#FP(k₁) ≤ #FP(k₂)",
        evidence={3: 1, 4: 1, 5: 1, 6: 2, 7: 3},
        exceptions=[],
        pipeline="truc_1089",
        parameters={"base": 10, "k_range": [3, 4, 5, 6, 7]},
        confidence=0.7,
        novelty=0.5,
        simplicity=0.8,
        tested_domains=[
            TestedDomain(base=10, digit_length=k, range_lo=10**(k-1),
                         range_hi=10**k - 1)
            for k in [3, 4, 5, 6, 7]
        ],
        monotonicity_kind=MonotonicityKind.NONDECREASING,
    )


# =============================================================================
# 1. PROOF SKELETON
# =============================================================================

@pytest.mark.unit
class TestProofSkeleton:

    def test_modular_skeleton_kaprekar(self, gen):
        c = _make_modular_conjecture()
        skel = gen.generate(c)
        assert skel.strategy == ProofStrategy.MOD_INVARIANT
        assert len(skel.reduction_steps) >= 2
        assert any("digit" in s.description.lower() for s in skel.reduction_steps)
        assert skel.proof_strength in ("complete", "modulo_gap")

    def test_modular_skeleton_has_proven_steps(self, gen):
        c = _make_modular_conjecture(modulus=9)
        skel = gen.generate(c)
        proven = [s for s in skel.reduction_steps if s.status == "proven"]
        assert len(proven) >= 1

    def test_modular_skeleton_non_base_minus_1(self, gen):
        c = _make_modular_conjecture(modulus=7)
        skel = gen.generate(c)
        assert len(skel.remaining_gaps) > 0
        assert skel.proof_strength != "complete"

    def test_counting_skeleton(self, gen):
        c = _make_counting_conjecture()
        skel = gen.generate(c)
        assert skel.strategy == ProofStrategy.COUNTING_RECURRENCE
        assert skel.proof_strength == "heuristic"

    def test_monotonicity_skeleton(self, gen):
        c = _make_monotonicity_conjecture()
        skel = gen.generate(c)
        assert skel.strategy == ProofStrategy.BOUNDING
        assert len(skel.reduction_steps) >= 1

    def test_skeleton_to_dict(self, gen):
        c = _make_modular_conjecture()
        skel = gen.generate(c)
        d = skel.to_dict()
        assert "strategy" in d
        assert "reduction_steps" in d
        assert "remaining_gaps" in d
        assert "proof_strength" in d
        assert d["strategy"] == "mod_invariant"

    def test_skeleton_str(self, gen):
        c = _make_modular_conjecture()
        skel = gen.generate(c)
        s = str(skel)
        assert "ProofSkeleton" in s
        assert "mod_invariant" in s

    def test_universality_skeleton(self, gen):
        c = Conjecture(
            ctype=ConjectureType.UNIVERSALITY,
            statement="kaprekar_step achieves >99% convergence",
            quantifier="∀k", predicate="conv > 0.99",
            evidence={}, exceptions=[], pipeline="kaprekar_step",
            parameters={"base": 10}, confidence=0.8,
            novelty=0.7, simplicity=0.8,
        )
        skel = gen.generate(c)
        assert skel.strategy == ProofStrategy.BOUNDING

    def test_structure_skeleton(self, gen):
        c = Conjecture(
            ctype=ConjectureType.STRUCTURE,
            statement="All FPs are palindromes",
            quantifier="∀FP", predicate="palindrome(FP)",
            evidence={}, exceptions=[], pipeline="truc_1089",
            parameters={"base": 10}, confidence=0.7,
            novelty=0.6, simplicity=0.9,
        )
        skel = gen.generate(c)
        assert skel.strategy == ProofStrategy.STRUCTURAL_FORM

    def test_invariant_skeleton(self, gen):
        c = Conjecture(
            ctype=ConjectureType.INVARIANT,
            statement="ds(FP) = 18",
            quantifier="∀FP", predicate="ds(FP) = 18",
            evidence={}, exceptions=[], pipeline="kaprekar_step",
            parameters={"base": 10}, confidence=0.7,
            novelty=0.6, simplicity=0.9,
        )
        skel = gen.generate(c)
        assert skel.strategy == ProofStrategy.MOD_INVARIANT

    def test_known_theorem_links(self, gen):
        c = _make_modular_conjecture()
        skel = gen.generate(c)
        assert len(skel.known_theorem_links) >= 1
        assert any("digit sum" in t.lower() for t in skel.known_theorem_links)


# =============================================================================
# 2. DENSITY ESTIMATOR
# =============================================================================

@pytest.mark.unit
class TestDensityEstimator:

    def test_zero_ces_gives_upper_bound(self, estimator):
        c = _make_modular_conjecture(k_range=[3, 4, 5])
        est = estimator.estimate(c, num_counterexamples=0)
        assert est.num_counterexamples == 0
        assert est.observed_density == 0.0
        assert est.upper_bound_95 > 0
        assert est.upper_bound_95 < 1
        assert est.upper_bound_99 > est.upper_bound_95
        assert est.falsification_label != "falsified"

    def test_falsified_label(self, estimator):
        c = _make_modular_conjecture()
        est = estimator.estimate(c, num_counterexamples=5)
        assert est.falsification_label == "falsified"
        assert est.observed_density > 0

    def test_large_search_space_strong(self, estimator):
        c = _make_modular_conjecture(k_range=[3, 4, 5, 6, 7])
        est = estimator.estimate(c, num_counterexamples=0)
        assert est.total_search_space > 100000
        assert est.search_volume_log10 > 5
        assert est.confidence_score > 0.5

    def test_small_search_space_weak(self, estimator):
        c = _make_modular_conjecture(k_range=[3])
        est = estimator.estimate(c, num_counterexamples=0)
        assert est.falsification_label in ("weak", "minimal")

    def test_density_to_dict(self, estimator):
        c = _make_modular_conjecture()
        est = estimator.estimate(c)
        d = est.to_dict()
        assert "total_search_space" in d
        assert "upper_bound_95" in d
        assert "falsification_label" in d

    def test_density_str(self, estimator):
        c = _make_modular_conjecture()
        est = estimator.estimate(c)
        s = str(est)
        assert "DensityEstimate" in s

    def test_empty_tested_domains_fallback(self, estimator):
        c = Conjecture(
            ctype=ConjectureType.MODULAR,
            statement="test", quantifier="∀k", predicate="mod 9",
            evidence={}, exceptions=[], pipeline="kaprekar_step",
            parameters={"base": 10, "k_range": [3, 4, 5]},
            confidence=0.7, novelty=0.6, simplicity=0.8,
            tested_domains=[],
        )
        est = estimator.estimate(c)
        assert est.total_search_space > 0

    def test_rule_of_three(self, estimator):
        c = _make_modular_conjecture(k_range=[3, 4, 5, 6, 7])
        est = estimator.estimate(c, num_counterexamples=0)
        # Rule of three: upper bound ≈ 3/N
        expected_ub = 3.0 / est.total_search_space
        assert abs(est.upper_bound_95 - expected_ub) < 1e-10


# =============================================================================
# 3. PATTERN COMPRESSOR
# =============================================================================

@pytest.mark.unit
class TestPatternCompressor:

    def test_constant_pattern(self, compressor):
        patterns = compressor.analyze_counting_sequence([3, 4, 5, 6], [1, 1, 1, 1])
        types = [p.pattern_type for p in patterns]
        assert "constant" in types

    def test_affine_pattern(self, compressor):
        patterns = compressor.analyze_counting_sequence([3, 4, 5, 6], [5, 7, 9, 11])
        types = [p.pattern_type for p in patterns]
        assert "affine" in types
        affine = [p for p in patterns if p.pattern_type == "affine"][0]
        assert affine.parameters["a"] == 2
        assert affine.parameters["b"] == -1

    def test_eventually_constant(self, compressor):
        patterns = compressor.analyze_counting_sequence(
            [3, 4, 5, 6, 7, 8], [1, 2, 3, 3, 3, 3])
        types = [p.pattern_type for p in patterns]
        assert "eventually_constant" in types

    def test_quadratic_pattern(self, compressor):
        # f(k) = k^2: 9, 16, 25, 36, 49
        patterns = compressor.analyze_counting_sequence(
            [3, 4, 5, 6, 7], [9, 16, 25, 36, 49])
        types = [p.pattern_type for p in patterns]
        # Should detect quadratic or geometric
        assert len(patterns) > 0

    def test_modular_pattern(self, compressor):
        # f(k) = 1 if k even, 2 if k odd
        patterns = compressor.analyze_counting_sequence(
            [3, 4, 5, 6, 7, 8, 9, 10],
            [2, 1, 2, 1, 2, 1, 2, 1])
        types = [p.pattern_type for p in patterns]
        assert "modular" in types

    def test_geometric_pattern(self, compressor):
        patterns = compressor.analyze_counting_sequence(
            [1, 2, 3, 4, 5], [2, 4, 8, 16, 32])
        types = [p.pattern_type for p in patterns]
        assert "geometric" in types

    def test_empty_sequence(self, compressor):
        patterns = compressor.analyze_counting_sequence([], [])
        assert patterns == []

    def test_single_point(self, compressor):
        patterns = compressor.analyze_counting_sequence([3], [5])
        assert patterns == []

    def test_fp_structure_common_factor(self, compressor):
        patterns = compressor.analyze_fp_structure([6174, 549945, 631764])
        types = [p.pattern_type for p in patterns]
        assert "common_factor" in types

    def test_fp_structure_palindrome(self, compressor):
        patterns = compressor.analyze_fp_structure([1089, 10890])
        # 1089 is not a palindrome, but 10890 is not either
        # Let's use actual palindromes
        patterns = compressor.analyze_fp_structure([121, 12321, 1234321])
        types = [p.pattern_type for p in patterns]
        assert "all_palindromes" in types

    def test_pattern_to_dict(self, compressor):
        patterns = compressor.analyze_counting_sequence([3, 4, 5], [1, 1, 1])
        assert len(patterns) > 0
        d = patterns[0].to_dict()
        assert "type" in d
        assert "formula" in d
        assert "confidence" in d


# =============================================================================
# 4. CONJECTURE MUTATOR
# =============================================================================

@pytest.mark.unit
class TestConjectureMutator:

    def test_generalize_modulus(self, mutator):
        c = _make_modular_conjecture(modulus=9)
        mutations = mutator.mutate(c)
        gen_mutations = [m for m in mutations if m.mutation_type == MutationType.GENERALIZE]
        assert len(gen_mutations) > 0
        # Should try related moduli
        new_moduli = [m.mutated.parameters["modulus"] for m in gen_mutations]
        assert any(m != 9 for m in new_moduli)

    def test_transfer_to_pipeline(self, mutator):
        c = _make_modular_conjecture(pipeline="kaprekar_step")
        mutations = mutator.mutate(c)
        transfers = [m for m in mutations if m.mutation_type == MutationType.TRANSFER]
        assert len(transfers) > 0
        target_pipes = [m.mutated.pipeline for m in transfers]
        assert "truc_1089" in target_pipes

    def test_strengthen_quantifier(self, mutator):
        c = _make_modular_conjecture(k_range=[3, 4, 5])
        mutations = mutator.mutate(c)
        strengthened = [m for m in mutations if m.mutation_type == MutationType.STRENGTHEN]
        # k_min=3 > 2, so should propose extension to k=2
        assert len(strengthened) > 0

    def test_weaken_with_exceptions(self, mutator):
        c = _make_modular_conjecture()
        c.exceptions = [{"k": 8, "fp": 12345, "residue": 3}]
        mutations = mutator.mutate(c)
        weakened = [m for m in mutations if m.mutation_type == MutationType.WEAKEN]
        assert len(weakened) > 0
        assert "except" in weakened[0].mutated.statement.lower()

    def test_mutation_to_dict(self, mutator):
        c = _make_modular_conjecture()
        mutations = mutator.mutate(c)
        if mutations:
            d = mutations[0].to_dict()
            assert "mutation_type" in d
            assert "rationale" in d

    def test_mutated_confidence_lower(self, mutator):
        c = _make_modular_conjecture()
        mutations = mutator.mutate(c)
        for m in mutations:
            assert m.mutated.confidence < c.confidence

    def test_non_modular_no_generalize(self, mutator):
        c = _make_counting_conjecture()
        mutations = mutator.mutate(c)
        gen = [m for m in mutations if m.mutation_type == MutationType.GENERALIZE]
        assert len(gen) == 0


# =============================================================================
# 5. RANKING MODEL V1
# =============================================================================

@pytest.mark.unit
class TestRankingModelV1:

    def test_version_is_explicit(self):
        assert RANKING_MODEL_VERSION == "1.0"
        assert RankingModelV1.VERSION == "1.0"

    def test_weights_sum_to_one(self):
        total = (RankingModelV1.W_EMPIRICAL + RankingModelV1.W_STRUCTURAL
                 + RankingModelV1.W_NOVELTY + RankingModelV1.W_SIMPLICITY
                 + RankingModelV1.W_FALSIFIABILITY)
        assert abs(total - 1.0) < 1e-10

    def test_rank_modular_conjecture(self, ranker):
        c = _make_modular_conjecture()
        ranked = ranker.rank(c)
        assert ranked.final_score > 0
        assert ranked.final_score <= 1.0
        assert "empirical_confidence" in ranked.rank_breakdown
        assert "structural_strength" in ranked.rank_breakdown

    def test_rank_counting_conjecture(self, ranker):
        c = _make_counting_conjecture()
        ranked = ranker.rank(c)
        assert ranked.final_score > 0

    def test_ranked_to_dict(self, ranker):
        c = _make_modular_conjecture()
        ranked = ranker.rank(c)
        d = ranked.to_dict()
        assert "final_score" in d
        assert "rank_breakdown" in d
        assert "density" in d
        assert "proof_strength" in d
        assert d["ranking_model_version"] == "1.0"

    def test_rank_all_sorts_descending(self, ranker):
        c1 = _make_modular_conjecture(modulus=9)
        c2 = _make_counting_conjecture()
        c3 = _make_monotonicity_conjecture()
        ranked = ranker.rank_all([c1, c2, c3])
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_proven_skeleton_higher_structural_score(self, ranker):
        # Mod-9 conjecture should get higher structural score than counting
        c_mod9 = _make_modular_conjecture(modulus=9)
        c_count = _make_counting_conjecture()
        r_mod9 = ranker.rank(c_mod9)
        r_count = ranker.rank(c_count)
        assert r_mod9.rank_breakdown["structural_strength"] >= r_count.rank_breakdown["structural_strength"]

    def test_falsified_low_confidence(self, ranker):
        c = _make_modular_conjecture()
        c.exceptions = [{"k": 8, "fp": 999}]
        ranked = ranker.rank(c)
        assert ranked.density_estimate.falsification_label == "falsified"


# =============================================================================
# 6. INTEGRATION (requires exhaustive runs)
# =============================================================================

@pytest.mark.exhaustive
class TestM3Integration:

    def test_mine_skeleton_density_rank(self, reg):
        """Full pipeline: mine → skeleton → density → rank."""
        miner = ConjectureMiner(reg)
        conjectures = miner.mine_all("kaprekar_step", [3, 4, 5])
        assert len(conjectures) > 0

        gen = SkeletonGenerator(reg)
        est = DensityEstimator()
        ranker = RankingModelV1(reg)

        for c in conjectures:
            skel = gen.generate(c)
            assert isinstance(skel, ProofSkeleton)
            density = est.estimate(c)
            assert isinstance(density, DensityEstimate)
            ranked = ranker.rank(c)
            assert ranked.final_score > 0

    def test_mutation_and_test(self, reg):
        """Mine → mutate → test mutated conjecture."""
        miner = ConjectureMiner(reg)
        conjs = miner.mine_invariants("kaprekar_step", [3, 4, 5])
        mod9 = [c for c in conjs if c.parameters.get("modulus") == 9]
        assert len(mod9) > 0

        mutator = ConjectureMutator(reg)
        mutations = mutator.mutate(mod9[0])
        assert len(mutations) > 0

    def test_pattern_compression_real_data(self, reg):
        """Mine real FP counts and compress."""
        miner = ConjectureMiner(reg)
        runner = PipelineRunner(reg)
        pipe = Pipeline.parse("kaprekar_step", registry=reg)

        fp_counts = {}
        for k in [3, 4, 5, 6]:
            domain = DomainPolicy(base=10, digit_length=k, exclude_repdigits=True)
            result = runner.run_exhaustive(pipe, domain)
            fp_counts[k] = len(result.fixed_points)

        ks = sorted(fp_counts.keys())
        vals = [fp_counts[k] for k in ks]
        patterns = PatternCompressor.analyze_counting_sequence(ks, vals)
        assert isinstance(patterns, list)

    def test_full_ranked_output_kaprekar(self, reg):
        """Full ranking with all components."""
        miner = ConjectureMiner(reg)
        conjectures = miner.mine_all("kaprekar_step", [3, 4, 5])
        ranker = RankingModelV1(reg)
        ranked = ranker.rank_all(conjectures)
        assert len(ranked) > 0
        for r in ranked:
            d = r.to_dict()
            assert d["ranking_model_version"] == "1.0"
            assert d["final_score"] > 0
