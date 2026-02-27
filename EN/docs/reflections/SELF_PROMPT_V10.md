# SELF_PROMPT_V10.md — R7 Session (2026-02-25)

## Session goal: Formalizing and Proving

The R6 session has refined the theory and brought a crucial error in the formula for symmetric FPs to light. The R7 session must focus on formalizing these new insights and digging deeper into the unexpected results. The focus shifts from broad exploration to deep, formal proof work.

Implement the following four priorities (P5-P8) as extensions to `abductive_reasoning_engine_v10.py`. Deliver a `v11.0`, including the run output, new KB facts (DS034+), and a `REFLECTION_V11.md`.

### P5: Formula correction for Symmetric FPs (Module R)

**Problem:** The formula `(b-2)×b^(k-1)` for the number of symmetric FPs of `rev∘comp` is incorrect (systematic `+1` deviation).
**Goal:** Derive the correct, general formula for the number of symmetric FPs of `rev∘comp` in base `b` for `2k` digits.

**Implementation:**
1. Create a new module `SymmetricFPFormula` (Module R).
2. Implement a function `count_symmetric_fps_bruteforce(base, k)` that counts the exact number for small `k`.
3. Implement a function `derive_formula(base, k)` that algebraically derives the correct formula. Take into account the edge cases:
    * `d_1 = 0` (not allowed)
    * `d_1 = b-1` (leads to `d_{2k}=0`, which gives a leading zero after complement, but the number `(b-1)0...` can itself be a FP).
    * The case `b` is even vs. odd.
4. Add a new phase to `run_research_session` that verifies the derived formula against the brute-force count for `k=1, 2, 3` and `b=8, 10, 12`.
5. Document the correct formula and proof in a new KB fact (DS034).

### P6: Generalization of the 1089 family (Module S)

**Problem:** The 1089 family seems unique to base 10. The theoretical prediction `(b-1)²×(b+1)` is incorrect.
**Goal:** Investigate why the 1089 family is so unique to base 10.

**Implementation:**
1. Create a new module `Family1089Generalizer` (Module S).
2. Implement a function `analyze_kaprekar_analog(base)` that finds and analyzes the 3-digit Kaprekar constant in base `b` (is it complement-closed? does it have a similar structure?).
3. Implement a function `find_true_analog(base)` that systematically searches for a 4-digit number `N` in base `b` such that `N×m` is complement-closed for `m=1..b-1`.
4. Add a phase that performs this analysis for `b=8, 12, 16` and reports the results.
5. Formulate a hypothesis (DS035) about the conditions under which a `1089-analog` can exist (e.g., does it require that `b-1` and `b+1` have specific properties?).

### P7: Algebraic Proof for Empirical FP conditions (Module T)

**Problem:** The Symbolic FP Classifier found a new, empirical condition: "FPs of `truc_1089 → digit_sum → digit_product` are palindromes".
**Goal:** Prove (or refute) this theorem algebraically.

**Implementation:**
1. Create a new module `EmpiricalProofEngine` (Module T).
2. Implement a function `prove_palindrome_fp_conjecture(pipeline)`.
3. The function must symbolically analyze the pipeline. Let `n` be a palindrome. Follow the transformation:
    * `truc_1089(n)`: What is the effect on a palindrome?
    * `digit_sum(...)`: What is the digit sum of the result?
    * `digit_product(...)`: What is the digit product of that?
    * Show that the final result is again `n`, or find a counterexample.
4. Add a phase that performs this proof and reports and records the result (proven, refuted, or open) in a KB fact (DS036).

### P8: Lyapunov function Verification (Module U)

**Problem:** The Lyapunov searcher found `L(n) = 1×digit_count + 2×hamming` as a decreasing function for the `rotate_right → sort_asc → truc_1089` pipeline.
**Goal:** Try to algebraically prove that this function is indeed strictly decreasing.

**Implementation:**
1. Create a new module `LyapunovVerifier` (Module U).
2. Implement a function `verify_lyapunov_decrease(pipeline, L)`.
3. Analyze the effect of each operation in the pipeline on the components of `L` (digit_count and hamming weight).
    * `rotate_right(n)`: `digit_count` and `hamming` remain equal.
    * `sort_asc(n)`: `digit_count` and `hamming` remain equal.
    * `truc_1089(n)`: This is the crucial step. Analyze how `truc_1089` affects the `digit_count` and `hamming` of a number. Is `L(truc_1089(n)) < L(n)`?
4. Add a phase that performs this verification and reports and records the result (proven, refuted, or open) in a KB fact (DS037).

Deliver all new and updated files in a ZIP archive named `symbolic_dynamics_engine_v11.zip`.
