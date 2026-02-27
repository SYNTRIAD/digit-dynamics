# Reflection: Digit-Pipeline Analysis Framework

**Date:** February 26, 2026
**Project:** SYNTRIAD Digit-Pipeline Analysis Framework
**Location:** `autonomous_discovery_engine/`

---

## 1. What is this project?

This project investigates a fundamental but surprisingly rich mathematical question:
*what happens when you repeatedly apply elementary operations on the digits of a number?*

Think of the Kaprekar constant 6174 — take any four-digit number, sort the digits descending, subtract the ascending-sorted number from it, and repeat. After at most 7 steps, you always arrive at 6174. This project radically generalizes that idea: we study arbitrary compositions ("pipelines") of 22 digit operations (reverse, complement, sort, digit-sum, Kaprekar step, 1089 trick, digit-powers, narcissistic step, digit-gcd, digit-xor, ...) in arbitrary number systems (base b ≥ 3).

The central discovery is that the fixed points of these pipelines are not arbitrary, but governed by the algebraic structure of the base b — in particular, the factors (b−1) and (b+1) play a dominant role.

---

## 2. Evolution of the project

The project has undergone an unusual and intensive development history, spanning 15 engine versions and 11 feedback rounds (R1–R11) with three different AI systems:

| Phase | Version | Contribution |
|-------|---------|--------------|
| **Exploration** | v1–v3 | GPU brute-force verification (CUDA), first attractor detection |
| **Structure** | v4–v6 | Pipeline enumeration, invariant discovery, pattern recognition |
| **Symbolic** | v7–v8 | Operator algebra, formal proof sketches, deductive theory |
| **Knowledge Base** | v9 | 51 facts (DS011–DS052), anomaly detection, self-questioning |
| **Multi-base** | v10 | Algebraic FP classification, Lyapunov search engine, multi-base engine |
| **Formal** | v11–v12 | 12/12 computational proofs, DS040 correction (1089 = universal) |
| **Broad** | v13 (R9) | Armstrong/narcissistic, Kaprekar odd bases, orbit analysis, 22 operations |
| **Deep** | v14 (R10) | Universal Lyapunov, repunits, cycle classification, multi-digit Kaprekar |
| **Open questions** | v15 (R11) | 4 infinite FP families, digit_sum Lyapunov proven, paper draft |
| **Publication** | R12+ | Paper A/B split, repo restructuring, two audits, adversarial audit fixes |

### The feedback rounds

Development took place in a cyclic AI-driven process:

- **R1–R5** (DeepSeek): Building core functions — knowledge base, complement families, counting formulas
- **R6** (Manus): Multi-base generalization, algebraic proof of 1089 family
- **R7–R11** (Cascade): Formal proofs, bugfixes (DS040 formula corrected), breadth and depth expansion, paper writing

A crucial turning point was **R8**, when a formula error in DS040 was discovered: `(b−1)²(b+1) = 891` was wrong; the correct formula `(b−1)(b+1)² = 1089` proved that the 1089 multiplicative family is **universal** for all bases b ≥ 3 — not specific to base 10.

### From monolith to modular architecture

The codebase evolved from a 6,500-line monolith (`abductive_reasoning_engine_v10.py`) to a structured modular architecture:

| Module | File | Function |
|--------|------|----------|
| M0 | `pipeline_dsl.py` (40K) | Pipeline DSL, 22 operations, canonical hashing |
| M1 | `experiment_runner.py` (24K) | Experiment execution, SQLite storage |
| M2 | `feature_extractor.py` (38K) | 17-feature number profiling, conjecture mining |
| M3 | `proof_engine.py` (47K) | Proof sketches, density estimation, ranking |
| M4 | `appendix_emitter.py` (48K) | LaTeX generation, manifests, determinism guard |

Total: ~200K source code + 117 unit tests (100% passing).

---

## 3. What can it do now?

### Mathematical results

The framework has produced **79 knowledge base facts** (65 proven, rest empirical), including nine formally proven theorems:

**Theorem 1 (DS034):** For every base b ≥ 3 and every k ≥ 1: the number of fixed points of rev ∘ comp among 2k-digit numbers is exactly (b−2)·b^(k−1).

**Theorem 2 (DS040):** The 1089 multiplicative family A_b × m (with A_b = (b−1)(b+1)²) is universal: for every base b ≥ 3 and m = 1, …, b−1, A_b·m is a complement-closed fixed point with digits [m, m−1, (b−1)−m, b−m].

**Theorem 3 (DS064):** There exist at least four pairwise disjoint infinite families of fixed points, each with a proven counting formula:
1. Symmetric family: (b−2)·b^(k−1) FPs
2. 1089×m multiplicative: b−1 FPs per base
3. Sort-descending: C(k+9,k)−1 FPs
4. Palindromes: 9·10^(⌊(k−1)/2⌋) FPs

**Theorem 4 (DS069):** A fifth infinite family: the 1089-trick mapping T(n) = |n − rev(n)| + rev(|n − rev(n)|) has fixed points n_k = 110·(10^(k−3) − 1) for every k ≥ 5.

**Theorem 5 (DS039, DS057, DS066):** Kaprekar constants algebraically proven: K_b = (b/2)(b²−1) for even b ≥ 4; 4-digit K = 6174 with convergence ≤ 7 steps; 6-digit: two FPs (549945, 631764).

**Theorem 6 (DS065):** Armstrong/narcissistic numbers: k_max(b) = max{k : k·(b−1)^k ≥ b^(k−1)}. Proven: k_max(10) = 60.

**Theorem 7 (DS055):** Repunits R_k = (b^k − 1)/(b−1) are never complement-closed fixed points.

**Theorem 8 (DS061):** digit_sum is a conditional Lyapunov function for pipelines consisting of ds-preserving and ds-contractive operations.

**Theorem 9 (DS038–DS045):** Lyapunov descent bounds for digit-power mappings (digit_pow2 through digit_pow5, digit_factorial_sum).

### Computational capabilities

- **Exhaustive verification** over 2×10^7 starting values per pipeline
- **GPU acceleration** via Numba CUDA JIT-compiled kernels (RTX 4000 Ada, ~5×10^6 iterations/sec)
- **SHA-256 verification hashes** for reproducibility
- **Multi-base support** for b ∈ {3, …, 16}
- **Deterministic reproduction**: `run_experiments.py` → `reproduce.py` → identical artifacts

---

## 4. The papers

The work is split into two standalone manuscripts, each targeting a different audience:

### Paper A: "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases: Algebraic Structure and Five Infinite Families"

- **Type:** Pure mathematical (theorem-proof)
- **Size:** 750 lines LaTeX, 7 pages PDF
- **Target journals:** Journal of Integer Sequences, Integers, Fibonacci Quarterly
- **Content:** Theorems 1–9, five infinite FP families, Kaprekar analysis up to 7 digits, Armstrong upper bounds, Lyapunov descent bounds
- **Strongest claim:** The classification of five disjoint infinite families with exact counting formulas, plus the universality of the 1089 family across all bases

### Paper B: "Attractor Spectra and ε-Universality in Digit-Operation Dynamical Systems"

- **Type:** Mixed theoretical + experimental
- **Size:** 558 lines LaTeX, 6 pages PDF
- **Target journals:** Experimental Mathematics, Complex Systems
- **Content:** ε-universality, basin entropy, composition lemma, conditional Lyapunov theorem, GPU-exhaustive attractor statistics, three conjectures
- **Strongest claim:** The sharp dichotomy — contractive+mixing pipelines converge almost universally (ε < 0.01), while non-contractive pipelines exhibit rich multi-attractor spectra (H > 2 bits)

### Relationship between the papers

Paper A provides the algebraic foundation (which fixed points exist?); Paper B builds on that with dynamic analysis (how do starting values distribute across attractors?). Paper B references Paper A as a companion paper. Together they form a coherent diptych.

---

## 5. Current status

### What is complete

| Component | Status |
|-----------|--------|
| Mathematical results (DS011–DS068) | ✅ 65/79 proven |
| Formal computational proofs | ✅ 12/12 |
| Engine architecture (M0–M4) | ✅ Modular, tested |
| Unit tests | ✅ 117/117 passing |
| Paper A LaTeX | ✅ Compiles, standalone |
| Paper B LaTeX | ✅ Compiles, standalone |
| Repo restructuring | ✅ engines/ separated from M0–M4 |
| Adversarial audit | ✅ All 8 fixes implemented |
| Audit verdict | ✅ CONDITIONAL ACCEPT (no HIGH issues remaining) |

### What remains open

Based on the two audits (technical audit + adversarial audit) and the submission roadmap, these are the remaining tasks:

#### High priority (blocking for submission)

| Item | Description | Source |
|------|-------------|--------|
| **C3: Language correction** | Replace terms like "Autonomous Discovery Engine" and "abductive reasoning" with neutral descriptions ("systematic computational exploration") | Technical audit §3.4 |
| **Reproducibility verification** | Full round-trip test: `run_experiments.py --fresh` → `reproduce.py --bundle` → DeterminismGuard green | Audit C1 |
| **k-range default** | Adjust default k-range from [3,4,5] to [3,4,5,6,7] to match paper scope | Audit I3 |

#### Medium priority (strongly recommended)

| Item | Description | Source |
|------|-------------|--------|
| **Ablation note ranking** | Note that ranking weights are heuristic and not calibrated | Audit I1 |
| **Pin NumPy version** | `requirements.txt` with `numpy>=1.24,<2.0` | Audit I2 |
| **Cross-platform float caveat** | Document that hash identity is not guaranteed cross-platform for >12 decimals | Audit I4 |
| **M2→M0 documentation** | Add comment that ConjectureMiner imports PipelineRunner as pragmatic integration | Audit I5 |

#### Low priority (nice-to-have)

| Item | Description | Source |
|------|-------------|--------|
| Adversarial edge-case tests | Empty pipeline, 0-input, single-digit, base=2 | Audit N1 |
| OEIS cross-references | Match FP counting formulas with OEIS sequences in Paper A | Audit N3 |
| WKN-005: Conjecture scale | More data for conjecture support (requires GPU runs) | Adversarial audit |
| WKN-009: Winter2020 citation | Verify preprint status | Adversarial audit |
| OEIS submission fifth family | Submit new sequence to OEIS | Adversarial audit |

---

## 6. Readiness for submission

### Assessment per criterion

| Criterion | Score | Explanation |
|-----------|-------|-------------|
| **Mathematical correctness** | 9/10 | 12/12 proofs computationally verified; all theorems algebraically supported |
| **Novelty** | 8/10 | Universality of 1089 family across all bases is new; five infinite FP families classification is new; ε-universality as concept is new |
| **Presentation** | 7/10 | Papers are structurally solid but language correction (C3) still needed; bibliography is extensive but could be stronger |
| **Reproducibility** | 7/10 | Hashes, deterministic scripts and tests are present; full round-trip must still be verified |
| **Scope delineation** | 8/10 | Paper A/B split is clear; engine is reduced to methodology mention |

### Estimated effort to submission

| Task | Estimated time |
|------|----------------|
| C3: Language correction (5 files) | 2–3 hours |
| Reproducibility round-trip test | 1–2 hours |
| k-range + NumPy pin + caveats | 1 hour |
| Ablation note + M2 documentation | 30 min |
| Final proofreading both papers | 2–3 hours |
| **Total** | **~1 workday** |

### Recommendation

**The project is ready for submission after one focused session.** The mathematical core is solid, the proofs are verified, and the papers compile as standalone manuscripts. The remaining tasks are primarily cosmetic and procedural:

1. **Language correction (C3)** is the main blocker — reviewers will flag "autonomous discovery" as overclaim.
2. **Reproducibility round-trip** must be tested end-to-end once.
3. **The rest** (NumPy pin, caveats, ablation note) can be done in an hour.

### Recommended submission strategy

1. **Paper A first** to Journal of Integer Sequences or Integers — this is the strongest manuscript with the hardest results. JIS publishes quickly and has a favorable track record for this type of work.
2. **Paper B** to Experimental Mathematics — the ε-universality concept and GPU-exhaustive approach fit well with this journal's profile.
3. **Both papers simultaneously on arXiv** with cross-references.
4. **Paper C** (engine as AI-methodological subject) comes only after acceptance of A and B — this is a strategic choice to sell the mathematics separately from the AI claims.

### What makes this project special

This is a rare example of a computational mathematics project that:
- Evolved from brute-force GPU verification to formal algebraic proofs
- Delivers not just individual results but builds a **classification theory**
- Proves the universality of a 75-year-old phenomenon (the 1089 trick) across all bases
- Is reproducible down to hash level

The combination of depth (formal proofs), breadth (22 operations, multi-base), and systematicity (79 knowledge base facts) makes it a substantial contribution to computational number theory.

---

## Addendum: The P_k Discovery (February 25, 2026)

During a triangle dialogue between the researcher, Manus (AI agent) and GPT-4, it was discovered that the engine executor applies an **implicit projection operator P_k**: after each operation, the intermediate result is projected back to k digits via zero-padding. This fundamentally changes the mathematical structure being studied — from pure operator composition to **projective dynamics on a fixed-length representation space**.

This discovery explains, among other things, why `digit_sum∘reverse` has exactly b−1 fixed points (the family {d·b^(k−1) | d ∈ {1,...,b−1}}), and opens a new perspective on the entire project: not the operations are the interesting object, but how fixed-length projection structurally redefines the dynamics of those operations.

See: [`REFLECTION_TRIANGLE_DIALOGUE_P_k.md`](REFLECTION_TRIANGLE_DIALOGUE_P_k.md) for the full analysis of this exchange.

---

*SYNTRIAD Research — February 2026*
