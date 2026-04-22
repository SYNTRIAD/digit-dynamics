#!/usr/bin/env python3
"""
Unit tests voor SYNTRIAD Abductive Reasoning Engine v14.0
=========================================================

Test coverage:
  - DigitOp: basis operaties
  - BaseNDigitOps (Module N): multi-base operaties
  - SymbolicFPClassifier (Module O): pipeline FP-condities
  - LyapunovSearch (Module P): dalende functies
  - FamilyProof1089 (Module Q): 1089-familie bewijs
  - FormalProofEngine (Module R): formele bewijsverificatie
  - KnowledgeBase: KB-feiten integriteit
"""

import sys
import unittest

# Import engine
from abductive_reasoning_engine_v10 import (
    DigitOp, OPERATIONS, BaseNDigitOps, MultiBaseAnalyzer,
    SymbolicFPClassifier, LyapunovSearch, FamilyProof1089,
    FormalProofEngine, KnowledgeBase, KnownFact, ProofLevel,
    NarcissisticAnalyzer, OddBaseKaprekarAnalyzer, OrbitAnalyzer,
    ExtendedPipelineAnalyzer, UniversalLyapunovSearch, RepunitAnalyzer,
    CycleTaxonomy, MultiDigitKaprekar,
    KaprekarAlgebraicAnalyzer, ThirdFamilySearcher,
    DigitSumLyapunovProof, ArmstrongBoundAnalyzer,
    load_r6_kb_facts, load_r7_kb_facts, load_r8_kb_facts, load_r9_kb_facts,
    load_r10_kb_facts, load_r11_kb_facts,
    factorize, factor_str, digit_entropy,
)


class TestDigitOp(unittest.TestCase):
    """Test basis digit-operaties."""

    def test_reverse(self):
        self.assertEqual(DigitOp.reverse(1234), 4321)
        self.assertEqual(DigitOp.reverse(100), 1)
        self.assertEqual(DigitOp.reverse(0), 0)

    def test_digit_sum(self):
        self.assertEqual(DigitOp.digit_sum(1234), 10)
        self.assertEqual(DigitOp.digit_sum(999), 27)
        self.assertEqual(DigitOp.digit_sum(0), 0)

    def test_complement_9(self):
        self.assertEqual(DigitOp.complement_9(1234), 8765)
        self.assertEqual(DigitOp.complement_9(9999), 0)
        # complement_9(0) = 9-0 = 9 (niet 0!)
        self.assertEqual(DigitOp.complement_9(0), 9)
        # d_1+d_1' = 9 paarvorming
        for n in [18, 27, 36, 45, 1089, 2178]:
            comp = DigitOp.complement_9(n)
            self.assertEqual(len(str(n)), len(str(comp)) if comp > 0 else 0,
                             f"complement van {n}={comp} verliest digits")

    def test_sort_asc(self):
        self.assertEqual(DigitOp.sort_asc(4321), 1234)
        self.assertEqual(DigitOp.sort_asc(3021), 123)  # leading zero stripped

    def test_sort_desc(self):
        self.assertEqual(DigitOp.sort_desc(1234), 4321)
        self.assertEqual(DigitOp.sort_desc(1001), 1100)

    def test_kaprekar_step(self):
        self.assertEqual(DigitOp.kaprekar_step(495), 495)
        self.assertEqual(DigitOp.kaprekar_step(6174), 6174)

    def test_truc_1089(self):
        # Klassieke 1089-truc: 3-digit getal met d1 > d3
        self.assertEqual(DigitOp.truc_1089(321), 1089)
        self.assertEqual(DigitOp.truc_1089(532), 1089)

    def test_rotate(self):
        self.assertEqual(DigitOp.rotate_left(1234), 2341)
        self.assertEqual(DigitOp.rotate_right(1234), 4123)

    def test_digit_pow2(self):
        self.assertEqual(DigitOp.digit_pow2(12), 1 + 4)  # 1²+2²=5
        self.assertEqual(DigitOp.digit_pow2(99), 81 + 81)  # 162

    def test_add_reverse(self):
        self.assertEqual(DigitOp.add_reverse(123), 123 + 321)

    def test_swap_ends(self):
        self.assertEqual(DigitOp.swap_ends(1234), 4231)


class TestBaseNDigitOps(unittest.TestCase):
    """Test multi-base digit operaties (Module N)."""

    def test_to_digits_base10(self):
        eng = BaseNDigitOps(10)
        self.assertEqual(eng.to_digits(1234), [1, 2, 3, 4])

    def test_to_digits_base8(self):
        eng = BaseNDigitOps(8)
        # 255 in base 8 = 377
        self.assertEqual(eng.to_digits(255), [3, 7, 7])

    def test_from_digits(self):
        eng = BaseNDigitOps(10)
        self.assertEqual(eng.from_digits([1, 2, 3, 4]), 1234)
        eng8 = BaseNDigitOps(8)
        self.assertEqual(eng8.from_digits([3, 7, 7]), 255)

    def test_complement_base10(self):
        eng = BaseNDigitOps(10)
        self.assertEqual(eng.complement(1234), 8765)

    def test_complement_base8(self):
        eng = BaseNDigitOps(8)
        # 255 = [3,7,7] in base 8, complement = [4,0,0] = 4*64 = 256
        self.assertEqual(eng.complement(255), 256)

    def test_reverse_base(self):
        eng = BaseNDigitOps(10)
        self.assertEqual(eng.reverse(1234), 4321)

    def test_is_symmetric_base10(self):
        eng = BaseNDigitOps(10)
        # 18: digits [1,8], 1+8=9 ✓
        self.assertTrue(eng.is_symmetric(18))
        # 19: digits [1,9], 1+9=10 ≠ 9
        self.assertFalse(eng.is_symmetric(19))

    def test_symmetric_fp_count(self):
        """Verifieer (b-2) FPs voor 2-digit getallen in diverse bases."""
        for b in [6, 8, 10, 12, 16]:
            eng = BaseNDigitOps(b)
            lo, hi = b, b * b
            count = 0
            for n in range(lo, hi):
                comp_n = eng.complement(n)
                rev_comp_n = eng.reverse(comp_n)
                if rev_comp_n == n:
                    count += 1
            self.assertEqual(count, b - 2,
                             f"Basis {b}: verwacht {b-2} FPs, gevonden {count}")


class TestSymbolicFPClassifier(unittest.TestCase):
    """Test algebraïsche FP-condities (Module O)."""

    def setUp(self):
        self.clf = SymbolicFPClassifier(OPERATIONS)

    def test_reverse_palindrome(self):
        """reverse FPs = palindromen"""
        cond = self.clf.KNOWN_CONDITIONS[('reverse',)]
        self.assertTrue(cond['test'](121))
        self.assertTrue(cond['test'](1221))
        self.assertFalse(cond['test'](123))

    def test_complement_no_fp(self):
        """complement_9 heeft geen non-triviale FPs"""
        cond = self.clf.KNOWN_CONDITIONS[('complement_9',)]
        self.assertTrue(cond['test'](0))
        self.assertFalse(cond['test'](45))

    def test_rev_comp_symmetric(self):
        """rev∘comp FPs: d_i + d_{2k+1-i} = 9, d_1 ≤ 8"""
        cond = self.clf.KNOWN_CONDITIONS[('reverse', 'complement_9')]
        self.assertTrue(cond['test'](18))  # 1+8=9
        self.assertTrue(cond['test'](1098))  # 1+8=9, 0+9=9
        self.assertFalse(cond['test'](19))
        self.assertFalse(cond['test'](123))  # odd length

    def test_sort_desc_sort_asc(self):
        """sort_desc∘sort_asc FPs: aflopende digits"""
        cond = self.clf.KNOWN_CONDITIONS[('sort_desc', 'sort_asc')]
        self.assertTrue(cond['test'](9876))
        self.assertTrue(cond['test'](9))
        self.assertFalse(cond['test'](1234))

    def test_sort_asc_sort_desc(self):
        """sort_asc∘sort_desc FPs: oplopende digits"""
        cond = self.clf.KNOWN_CONDITIONS[('sort_asc', 'sort_desc')]
        self.assertTrue(cond['test'](1234))
        self.assertTrue(cond['test'](1))
        self.assertFalse(cond['test'](4321))

    def test_double_complement(self):
        """comp∘comp = id (elke n is FP)"""
        cond = self.clf.KNOWN_CONDITIONS[('complement_9', 'complement_9')]
        self.assertTrue(cond['test'](42))
        self.assertTrue(cond['test'](9999))

    def test_double_reverse(self):
        """rev∘rev = id voor n zonder trailing zeros"""
        cond = self.clf.KNOWN_CONDITIONS[('reverse', 'reverse')]
        self.assertTrue(cond['test'](123))
        self.assertFalse(cond['test'](120))  # trailing zero

    def test_digit_sum_fp(self):
        """digit_sum FPs: 0-9"""
        cond = self.clf.KNOWN_CONDITIONS[('digit_sum',)]
        for n in range(10):
            self.assertTrue(cond['test'](n))
        self.assertFalse(cond['test'](10))

    def test_kaprekar_fp(self):
        """kaprekar_step FPs: 495, 6174"""
        cond = self.clf.KNOWN_CONDITIONS[('kaprekar_step',)]
        self.assertTrue(cond['test'](495))
        self.assertTrue(cond['test'](6174))
        self.assertFalse(cond['test'](1234))


class TestNewDigitOps(unittest.TestCase):
    """Test nieuwe digit-operaties (B4)."""

    def test_digit_gcd(self):
        self.assertEqual(DigitOp.digit_gcd(369), 3)  # gcd(3,6,9)=3
        self.assertEqual(DigitOp.digit_gcd(248), 2)  # gcd(2,4,8)=2
        self.assertEqual(DigitOp.digit_gcd(135), 1)  # gcd(1,3,5)=1
        self.assertEqual(DigitOp.digit_gcd(100), 1)  # gcd(1)=1, zeros skipped

    def test_digit_xor(self):
        self.assertEqual(DigitOp.digit_xor(123), 1 ^ 2 ^ 3)  # = 0
        self.assertEqual(DigitOp.digit_xor(999), 9 ^ 9 ^ 9)  # = 9
        self.assertEqual(DigitOp.digit_xor(0), 0)

    def test_narcissistic_step(self):
        # 153 = 1^3 + 5^3 + 3^3 (Armstrong number k=3)
        self.assertEqual(DigitOp.narcissistic_step(153), 153)
        # 370 = 3^3 + 7^3 + 0^3
        self.assertEqual(DigitOp.narcissistic_step(370), 370)
        # 9474 = 9^4 + 4^4 + 7^4 + 4^4 (Armstrong k=4)
        self.assertEqual(DigitOp.narcissistic_step(9474), 9474)
        # Non-Armstrong
        self.assertNotEqual(DigitOp.narcissistic_step(100), 100)

    def test_operations_count(self):
        """Moeten nu 22 operaties zijn."""
        self.assertEqual(len(OPERATIONS), 22)
        self.assertIn('digit_gcd', OPERATIONS)
        self.assertIn('digit_xor', OPERATIONS)
        self.assertIn('narcissistic_step', OPERATIONS)


class TestNarcissisticAnalyzer(unittest.TestCase):
    """Test Module S: Armstrong numbers en bifurcatie."""

    def setUp(self):
        self.analyzer = NarcissisticAnalyzer()

    def test_armstrong_k3(self):
        """DS047: Armstrong k=3 = {153, 370, 371, 407}"""
        result = self.analyzer.find_armstrong_numbers(3)
        self.assertEqual(result, [153, 370, 371, 407])

    def test_armstrong_k4(self):
        """DS048: Armstrong k=4 = {1634, 8208, 9474}"""
        result = self.analyzer.find_armstrong_numbers(4)
        self.assertEqual(result, [1634, 8208, 9474])

    def test_armstrong_k2_empty(self):
        """Geen Armstrong numbers met 2 digits."""
        result = self.analyzer.find_armstrong_numbers(2)
        self.assertEqual(result, [])

    def test_armstrong_k1(self):
        """Alle 1-digit getallen zijn Armstrong."""
        result = self.analyzer.find_armstrong_numbers(1)
        self.assertEqual(result, list(range(1, 10)))

    def test_bifurcation(self):
        """Bifurcatie-analyse moet correcte counts geven."""
        results = self.analyzer.bifurcation_analysis(max_k=4)
        self.assertEqual(results[3]['count'], 4)
        self.assertEqual(results[4]['count'], 3)
        self.assertFalse(results[1]['descent_proven'])  # k=1: 9 > 1


class TestOddBaseKaprekar(unittest.TestCase):
    """Test Module T: Kaprekar-dynamica in diverse bases."""

    def setUp(self):
        self.analyzer = OddBaseKaprekarAnalyzer()

    def test_base10_kaprekar(self):
        """Base 10: 495 moet een FP zijn."""
        result = self.analyzer.analyze_base(10, num_digits=3)
        self.assertIn(495, result['fixed_points'])

    def test_even_base_has_fp(self):
        """Even bases moeten een FP hebben (DS039/DS049)."""
        for b in [8, 10, 12]:
            result = self.analyzer.analyze_base(b, num_digits=3)
            self.assertGreater(result['num_fps'], 0,
                               f"Basis {b}: geen FP gevonden")

    def test_kaprekar_orbit_fp(self):
        """495 moet een FP-orbit hebben in basis 10."""
        result = self.analyzer.kaprekar_orbit(495, 10)
        self.assertEqual(result['type'], 'fixed_point')
        self.assertEqual(result['fixed_point'], 495)

    def test_classify_all(self):
        """Classificatie voor meerdere bases moet werken."""
        results = self.analyzer.classify_all_bases(bases=[8, 10], num_digits=3)
        self.assertIn(8, results)
        self.assertIn(10, results)


class TestOrbitAnalyzer(unittest.TestCase):
    """Test Module U: orbit-analyse."""

    def setUp(self):
        self.analyzer = OrbitAnalyzer(OPERATIONS)

    def test_kaprekar_orbit_to_495(self):
        """Kaprekar-orbit van 321 moet naar 495 convergeren."""
        result = self.analyzer.compute_orbit(321, ('kaprekar_step',))
        self.assertEqual(result['type'], 'fixed_point')
        self.assertEqual(result['value'], 495)

    def test_digit_sum_orbit(self):
        """digit_sum orbit convergeert naar single digit."""
        result = self.analyzer.compute_orbit(9999, ('digit_sum',))
        self.assertEqual(result['type'], 'fixed_point')
        self.assertLess(result['value'], 10)

    def test_pipeline_analysis(self):
        """Analyse van kaprekar_step moet convergent zijn."""
        result = self.analyzer.analyze_pipeline(
            ('kaprekar_step',), sample_size=50,
            domain=(100, 999))
        self.assertGreater(result['convergent_rate'], 0.5)


class TestExtendedPipeline(unittest.TestCase):
    """Test Module V: langere pipelines."""

    def setUp(self):
        self.analyzer = ExtendedPipelineAnalyzer(OPERATIONS)

    def test_generate_long_pipelines(self):
        pipes = self.analyzer.generate_long_pipelines(length=5, count=20)
        self.assertGreaterEqual(len(pipes), 10)
        for p in pipes:
            self.assertEqual(len(p), 5)

    def test_find_fps(self):
        """Kaprekar pipeline moet FPs vinden."""
        fps = self.analyzer.find_fps(('kaprekar_step',))
        self.assertIn(495, fps)

    def test_analyze_long_pipelines(self):
        results = self.analyzer.analyze_long_pipelines(lengths=[5], count_per_length=20)
        self.assertIn(5, results)
        self.assertGreater(results[5]['num_pipelines'], 0)


class TestUniversalLyapunov(unittest.TestCase):
    """Test Module W: universele Lyapunov-zoektocht."""

    def setUp(self):
        self.search = UniversalLyapunovSearch(OPERATIONS)

    def test_candidates_exist(self):
        self.assertIn('digit_sum', self.search.candidates)
        self.assertIn('log_n', self.search.candidates)
        self.assertGreaterEqual(len(self.search.candidates), 5)

    def test_test_candidate(self):
        result = self.search.test_candidate(
            'digit_sum', self.search.candidates['digit_sum'],
            ('digit_sum',), sample_size=50)
        self.assertIn('decrease_rate', result)
        self.assertGreaterEqual(result['decrease_rate'], 0)

    def test_search_universal(self):
        pipes = [('digit_sum',), ('kaprekar_step',)]
        result = self.search.search_universal(pipes, top_n=3)
        self.assertIn('ranked', result)
        self.assertIn('best', result)
        self.assertIsNotNone(result['best'])


class TestRepunitAnalyzer(unittest.TestCase):
    """Test Module X: repunit-analyse."""

    def setUp(self):
        self.analyzer = RepunitAnalyzer()

    def test_repunit_values(self):
        self.assertEqual(self.analyzer.repunit(1), 1)
        self.assertEqual(self.analyzer.repunit(2), 11)
        self.assertEqual(self.analyzer.repunit(3), 111)
        self.assertEqual(self.analyzer.repunit(4), 1111)

    def test_is_repunit(self):
        self.assertTrue(self.analyzer.is_repunit(111))
        self.assertFalse(self.analyzer.is_repunit(112))
        self.assertTrue(self.analyzer.is_repunit(1))

    def test_repunit_not_cc(self):
        """DS055: repunits zijn nooit CC FPs."""
        results = self.analyzer.repunit_properties(max_k=6)
        for r in results:
            self.assertFalse(r['is_cc_fp'],
                             f"R_{r['k']}={r['repunit']} zou geen CC FP moeten zijn")

    def test_repunit_base8(self):
        """Repunit in basis 8: (8^k-1)/7."""
        self.assertEqual(self.analyzer.repunit(3, base=8), 73)  # 1+8+64

    def test_cross_base(self):
        results = self.analyzer.cross_base_repunits(bases=[8, 10], max_k=4)
        self.assertIn(8, results)
        self.assertIn(10, results)


class TestCycleTaxonomy(unittest.TestCase):
    """Test Module Y: cyclus-taxonomie."""

    def setUp(self):
        self.tax = CycleTaxonomy(OPERATIONS)

    def test_classify_fp(self):
        """495 onder kaprekar_step moet FP zijn."""
        result = self.tax.classify_orbit(495, ('kaprekar_step',))
        self.assertEqual(result['class'], 'fp')
        self.assertEqual(result['attractor'], 495)

    def test_classify_divergent(self):
        """add_reverse kan divergeren."""
        result = self.tax.classify_orbit(99, ('add_reverse',), max_iter=50)
        # Can be fp, cycle, or divergent depending on number
        self.assertIn(result['class'], ['fp', 'divergent', 'unknown',
                                         '2-cycle', '3-cycle', '4-cycle'])

    def test_build_taxonomy(self):
        tax = self.tax.build_taxonomy(('kaprekar_step',),
                                      domain=(100, 999), sample_size=50)
        self.assertGreater(tax['sample_size'], 0)
        self.assertIn('class_distribution', tax)
        self.assertGreater(tax['fp_rate'], 0.5)


class TestMultiDigitKaprekar(unittest.TestCase):
    """Test Module Z: multi-digit Kaprekar."""

    def setUp(self):
        self.kap = MultiDigitKaprekar()

    def test_kaprekar_3digit_495(self):
        """3-digit Kaprekar moet naar 495 convergeren."""
        result = self.kap.kaprekar_orbit(321, 3)
        self.assertEqual(result['type'], 'fixed_point')
        self.assertEqual(result['value'], 495)

    def test_kaprekar_4digit_6174(self):
        """DS057: 4-digit Kaprekar = 6174."""
        result = self.kap.kaprekar_orbit(1234, 4)
        self.assertEqual(result['type'], 'fixed_point')
        self.assertEqual(result['value'], 6174)

    def test_kaprekar_4digit_max_7_steps(self):
        """4-digit convergentie in ≤7 stappen."""
        result = self.kap.kaprekar_orbit(9998, 4)
        self.assertEqual(result['type'], 'fixed_point')
        self.assertLessEqual(result['orbit_length'], 8)

    def test_analyze_digits(self):
        result = self.kap.analyze_digits(4, sample_size=100)
        self.assertIn(6174, result['fixed_points'])
        self.assertEqual(result['num_fps'], 1)

    def test_full_analysis(self):
        results = self.kap.full_analysis(digit_range=[3, 4], bases=[10])
        self.assertIn(10, results)
        self.assertIn(3, results[10])
        self.assertIn(4, results[10])


class TestFamilyProof1089(unittest.TestCase):
    """Test 1089-familie bewijs (Module Q)."""

    def setUp(self):
        self.proof = FamilyProof1089()

    def test_all_1089m_complement_closed(self):
        """Alle 1089×m (m=1..9) moeten complement-gesloten zijn."""
        results = self.proof.verify_complement_closed()
        for m in range(1, 10):
            self.assertTrue(results[m]['complement_closed'],
                            f"1089×{m}={1089*m} niet complement-gesloten")

    def test_digit_formula(self):
        """Digit-formule: 1089×m = [m, m-1, 9-m, 10-m]"""
        results = self.proof.verify_digit_formula()
        for m in range(1, 10):
            self.assertTrue(results[m]['match'],
                            f"Digit-formule klopt niet voor m={m}")

    def test_1089_factors(self):
        """1089 = 3² × 11²"""
        f = factorize(1089)
        self.assertEqual(f, {3: 2, 11: 2})


class TestFormalProofEngine(unittest.TestCase):
    """Test Module R: formele bewijsverificatie."""

    def setUp(self):
        self.engine = FormalProofEngine()

    def test_ds034_symmetric_formula(self):
        """DS034: (b-2)×b^(k-1) voor k=1 in bases 6,8,10."""
        results = self.engine.verify_symmetric_fp_formula(
            bases=[6, 8, 10], max_k=1)
        for b in [6, 8, 10]:
            self.assertTrue(results[b][1]['match'],
                            f"DS034 faalt voor b={b}, k=1")

    def test_ds035_divisibility(self):
        """DS035: CC getallen deelbaar door (b-1)."""
        results = self.engine.verify_complement_closed_divisibility(
            bases=[10], max_val=5000)
        self.assertTrue(results[10]['proven'])

    def test_ds036_complement_involution(self):
        """DS036: comp∘comp = id voor d_1 ≤ 8."""
        for n in [1, 18, 45, 1089, 8765]:
            self.assertEqual(
                DigitOp.complement_9(DigitOp.complement_9(n)), n,
                f"comp∘comp({n}) ≠ {n}")

    def test_ds036_leading_zero_exception(self):
        """DS036 uitzondering: d_1=9 breekt involutie."""
        # 90 → comp=09=9 → comp=0 ≠ 90
        self.assertNotEqual(
            DigitOp.complement_9(DigitOp.complement_9(90)), 90)

    def test_ds037_reverse_involution(self):
        """DS037: rev∘rev = id voor n zonder trailing zeros."""
        for n in [1, 12, 123, 1234, 9876]:
            self.assertEqual(
                DigitOp.reverse(DigitOp.reverse(n)), n,
                f"rev∘rev({n}) ≠ {n}")

    def test_ds037_trailing_zero_exception(self):
        """DS037 uitzondering: trailing zeros breken involutie."""
        self.assertNotEqual(DigitOp.reverse(DigitOp.reverse(120)), 120)

    def test_ds038_lyapunov(self):
        """DS038: digit_pow2(n) < n voor n ≥ 1000."""
        for n in [1000, 5000, 9999, 12345, 99999]:
            self.assertLess(DigitOp.digit_pow2(n), n,
                            f"digit_pow2({n}) niet < {n}")

    def test_ds039_kaprekar_constant(self):
        """DS039: K_b = (b/2)(b²-1) is FP van kaprekar_step voor even b."""
        results = self.engine.verify_kaprekar_constant(even_bases=[8, 10, 12, 16])
        for b in [8, 10, 12, 16]:
            self.assertTrue(results[b]['proven'],
                            f"DS039 faalt voor b={b}")
        # Specifieke waarden
        self.assertEqual(results[10]['K_b'], 495)
        self.assertEqual(results[8]['K_b'], 252)

    def test_ds040_1089_universal(self):
        """DS040: (b-1)(b+1)²×m is CC voor alle m=1..b-1 in diverse bases."""
        results = self.engine.verify_1089_universal(bases=[8, 10, 12])
        for b in [8, 10, 12]:
            self.assertTrue(results[b]['all_cc'],
                            f"DS040 faalt voor b={b}")
        # Base 10: A_b = 1089
        self.assertEqual(results[10]['A_b'], 1089)

    def test_ds041_odd_length_no_fps(self):
        """DS041: geen odd-length FPs van rev∘comp in even bases."""
        results = self.engine.verify_odd_length_no_fps(even_bases=[10])
        self.assertTrue(results[10]['proven'],
                        "DS041: odd-length FPs gevonden in basis 10")

    def test_ds042_lyapunov_pow3(self):
        """DS042: digit_pow3(n) < n voor n ≥ 10000."""
        for n in [10000, 50000, 99999]:
            self.assertLess(DigitOp.digit_pow3(n), n,
                            f"digit_pow3({n}) niet < {n}")

    def test_ds043_lyapunov_pow4(self):
        """DS043: digit_pow4(n) < n voor n ≥ 100000."""
        for n in [100000, 500000, 999999]:
            self.assertLess(DigitOp.digit_pow4(n), n,
                            f"digit_pow4({n}) niet < {n}")

    def test_ds044_lyapunov_pow5(self):
        """DS044: digit_pow5(n) < n voor n ≥ 1000000."""
        for n in [1000000, 5000000, 9999999]:
            self.assertLess(DigitOp.digit_pow5(n), n,
                            f"digit_pow5({n}) niet < {n}")

    def test_ds045_lyapunov_factorial(self):
        """DS045: digit_factorial_sum(n) < n voor n ≥ 10000000."""
        for n in [10000000, 50000000, 99999999]:
            self.assertLess(DigitOp.digit_factorial_sum(n), n,
                            f"digit_factorial_sum({n}) niet < {n}")


class TestKnowledgeBase(unittest.TestCase):
    """Test KB-integriteit."""

    def setUp(self):
        self.kb = KnowledgeBase()
        load_r6_kb_facts(self.kb)
        load_r7_kb_facts(self.kb)
        load_r8_kb_facts(self.kb)
        load_r9_kb_facts(self.kb)
        load_r10_kb_facts(self.kb)

    def test_all_facts_loaded(self):
        """Moeten minstens 60 feiten zijn (NT+OP+DS001-DS060)."""
        self.assertGreaterEqual(len(self.kb.facts), 60)

    def test_ds_ids_unique(self):
        """Alle fact-IDs moeten uniek zijn."""
        ids = list(self.kb.facts.keys())
        self.assertEqual(len(ids), len(set(ids)))

    def test_proven_facts_have_proofs(self):
        """Bewezen feiten moeten een niet-leeg bewijs hebben."""
        for fid, fact in self.kb.facts.items():
            if fact.proof_level == ProofLevel.PROVEN:
                self.assertTrue(len(fact.proof) > 10,
                                f"{fid} heeft geen bewijs")

    def test_ds020_formula(self):
        """DS020 moet de correcte formule 8×10^(k-1) bevatten."""
        ds020 = self.kb.facts.get("DS020")
        self.assertIsNotNone(ds020)
        self.assertIn("8", ds020.statement)

    def test_ds034_exists(self):
        """DS034 (formeel bewijs symmetrische formule) moet bestaan."""
        self.assertIn("DS034", self.kb.facts)
        self.assertEqual(self.kb.facts["DS034"].proof_level, ProofLevel.PROVEN)

    def test_r7_facts_loaded(self):
        """R7 feiten DS034-DS040 moeten allemaal geladen zijn."""
        for dsid in ["DS034", "DS035", "DS036", "DS037", "DS038", "DS039", "DS040"]:
            self.assertIn(dsid, self.kb.facts, f"{dsid} niet in KB")

    def test_r8_facts_loaded(self):
        """R8 feiten DS041-DS045 moeten allemaal geladen zijn."""
        for dsid in ["DS041", "DS042", "DS043", "DS044", "DS045"]:
            self.assertIn(dsid, self.kb.facts, f"{dsid} niet in KB")
            self.assertEqual(self.kb.facts[dsid].proof_level, ProofLevel.PROVEN,
                             f"{dsid} niet PROVEN")

    def test_ds039_is_proven(self):
        """DS039 moet PROVEN zijn (was EMPIRICAL in v11)."""
        self.assertEqual(self.kb.facts["DS039"].proof_level, ProofLevel.PROVEN)

    def test_ds040_is_universal(self):
        """DS040 moet 'UNIVERSEEL' bevatten (gecorrigeerd in R8)."""
        self.assertIn("UNIVERSEEL", self.kb.facts["DS040"].statement)

    def test_r9_facts_loaded(self):
        """R9 feiten DS046-DS052 moeten allemaal geladen zijn."""
        for dsid in ["DS046", "DS047", "DS048", "DS049", "DS050", "DS051", "DS052"]:
            self.assertIn(dsid, self.kb.facts, f"{dsid} niet in KB")

    def test_ds046_armstrong_finite(self):
        """DS046 moet PROVEN zijn."""
        self.assertEqual(self.kb.facts["DS046"].proof_level, ProofLevel.PROVEN)

    def test_ds052_odd_length_odd_base(self):
        """DS052 moet PROVEN zijn."""
        self.assertEqual(self.kb.facts["DS052"].proof_level, ProofLevel.PROVEN)

    def test_r10_facts_loaded(self):
        """R10 feiten DS053-DS060 moeten allemaal geladen zijn."""
        for dsid in ["DS053", "DS054", "DS055", "DS056", "DS057", "DS058", "DS059", "DS060"]:
            self.assertIn(dsid, self.kb.facts, f"{dsid} niet in KB")

    def test_ds055_repunit_proven(self):
        """DS055 (repunits niet CC) moet PROVEN zijn."""
        self.assertEqual(self.kb.facts["DS055"].proof_level, ProofLevel.PROVEN)

    def test_ds057_kaprekar_4digit(self):
        """DS057 (Kaprekar 4-digit = 6174) moet PROVEN zijn."""
        self.assertEqual(self.kb.facts["DS057"].proof_level, ProofLevel.PROVEN)


class TestKaprekarAlgebraic(unittest.TestCase):
    """R11: Kaprekar algebraïsche analyse."""

    def setUp(self):
        self.ka = KaprekarAlgebraicAnalyzer()

    def test_kaprekar_3digit_exhaustive(self):
        """Exhaustieve 3-digit Kaprekar moet 495 geven."""
        r = self.ka.exhaustive_kaprekar(3, 10)
        self.assertEqual(r['fixed_points'], [495])
        self.assertEqual(r['num_cycles'], 0)

    def test_kaprekar_4digit_exhaustive(self):
        """Exhaustieve 4-digit Kaprekar moet 6174 geven."""
        r = self.ka.exhaustive_kaprekar(4, 10)
        self.assertEqual(r['fixed_points'], [6174])
        self.assertLessEqual(r['max_convergence_steps'], 7)

    def test_kaprekar_6digit_two_fps(self):
        """DS066: 6-digit Kaprekar heeft 2 FPs."""
        r = self.ka.exhaustive_kaprekar(6, 10)
        self.assertIn(549945, r['fixed_points'])
        self.assertIn(631764, r['fixed_points'])
        self.assertEqual(r['num_fps'], 2)

    def test_kaprekar_fps_divisible_by_9(self):
        """DS067: alle Kaprekar FPs deelbaar door 9."""
        for d in [3, 4, 6]:
            r = self.ka.exhaustive_kaprekar(d, 10)
            for fp in r['fixed_points']:
                self.assertEqual(fp % 9, 0, f"Kaprekar FP {fp} (d={d}) not div by 9")

    def test_549945_is_palindrome(self):
        """549945 is een palindroom."""
        self.assertEqual(str(549945), str(549945)[::-1])


class TestThirdFamily(unittest.TestCase):
    """R11: Zoektocht naar 3e+ oneindige FP-familie."""

    def setUp(self):
        self.tf = ThirdFamilySearcher(OPERATIONS)

    def test_sort_desc_formula(self):
        """DS062: sort_desc FP-telling = C(k+9,k)-1 voor k>=2."""
        sf = self.tf.find_sort_desc_family(max_digits=4)
        self.assertTrue(sf['formula_match'])

    def test_sort_desc_is_infinite(self):
        """sort_desc familie is oneindig."""
        sf = self.tf.find_sort_desc_family(max_digits=3)
        self.assertTrue(sf['is_infinite'])

    def test_palindrome_family(self):
        """DS063: palindromen vormen oneindige FP-familie."""
        pf = self.tf.find_palindrome_family(max_digits=4)
        self.assertTrue(pf['is_infinite'])
        self.assertEqual(pf['counts'][2], 9)   # 11,22,...,99
        self.assertEqual(pf['counts'][3], 90)  # 101,111,...,999


class TestDigitSumLyapunov(unittest.TestCase):
    """R11: digit_sum Lyapunov bewijs."""

    def setUp(self):
        self.dl = DigitSumLyapunovProof(OPERATIONS)

    def test_not_universal(self):
        """DS061: digit_sum is NIET universeel Lyapunov."""
        cls = self.dl.classify_all_ops()
        proof = self.dl.prove_digit_sum_lyapunov(cls)
        self.assertFalse(proof['is_universal'])
        self.assertTrue(proof['is_conditional'])

    def test_sort_reverse_monotone(self):
        """sort en reverse behouden digit_sum (monotoon)."""
        cls = self.dl.classify_all_ops()
        for op in ['sort_asc', 'sort_desc', 'reverse']:
            self.assertTrue(cls[op]['is_monotone'],
                            f"{op} should be ds-monotone")

    def test_complement_not_monotone(self):
        """complement_9 is NIET ds-monotoon."""
        cls = self.dl.classify_all_ops()
        self.assertFalse(cls['complement_9']['is_monotone'])


class TestArmstrongBounds(unittest.TestCase):
    """R11: Armstrong k_max bounds."""

    def setUp(self):
        self.ab = ArmstrongBoundAnalyzer()

    def test_k_max_base10(self):
        """DS065: k_max(10) = 60."""
        r = self.ab.compute_k_max_bound(10)
        self.assertEqual(r['k_max'], 60)

    def test_k_max_base2(self):
        """k_max(2) moet klein zijn."""
        r = self.ab.compute_k_max_bound(2)
        self.assertLessEqual(r['k_max'], 2)

    def test_k_max_increases_with_base(self):
        """k_max moet toenemen met basis."""
        km = self.ab.k_max_cross_base([3, 5, 10, 16])
        vals = list(km['k_max_per_base'].values())
        for i in range(len(vals) - 1):
            self.assertLess(vals[i], vals[i+1])


class TestR11KBFacts(unittest.TestCase):
    """R11 KB-feiten DS061-DS068."""

    def setUp(self):
        self.kb = KnowledgeBase()
        load_r6_kb_facts(self.kb)
        load_r7_kb_facts(self.kb)
        load_r8_kb_facts(self.kb)
        load_r9_kb_facts(self.kb)
        load_r10_kb_facts(self.kb)
        load_r11_kb_facts(self.kb)

    def test_r11_facts_loaded(self):
        """R11 feiten DS061-DS068 moeten allemaal geladen zijn."""
        for i in range(61, 69):
            fid = f"DS0{i}"
            self.assertIn(fid, self.kb.facts, f"{fid} missing")

    def test_ds061_conditional_lyapunov(self):
        """DS061 moet PROVEN zijn."""
        self.assertEqual(self.kb.facts['DS061'].proof_level, ProofLevel.PROVEN)

    def test_ds064_four_families(self):
        """DS064 moet over 4 families gaan."""
        self.assertIn('4', self.kb.facts['DS064'].statement)

    def test_ds065_proven(self):
        """DS065 (k_max formule) moet PROVEN zijn."""
        self.assertEqual(self.kb.facts['DS065'].proof_level, ProofLevel.PROVEN)

    def test_total_facts_at_least_79(self):
        """Moeten minstens 79 feiten zijn."""
        self.assertGreaterEqual(len(self.kb.facts), 79)


class TestUtilities(unittest.TestCase):
    """Test utility functies."""

    def test_factorize(self):
        self.assertEqual(factorize(12), {2: 2, 3: 1})
        self.assertEqual(factorize(1089), {3: 2, 11: 2})
        self.assertEqual(factorize(1), {})

    def test_factor_str(self):
        self.assertEqual(factor_str(12), "2^2 * 3")

    def test_digit_entropy(self):
        self.assertAlmostEqual(digit_entropy(1111), 0.0)
        self.assertGreater(digit_entropy(1234), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
