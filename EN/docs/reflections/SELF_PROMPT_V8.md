# SELF-PROMPT: Symbolic Dynamics Engine v8.0
## From Symbolic Detection to Deductive Theory Generation

**Written by:** Cascade (AI pair programmer)
**Date:** 2026-02-23
**Context:** After v7.0 session with 300 pipelines, 109 unique attractors, 183 fixed points,
4 confirmed meta-theorems, 1 falsified, 100% symbolic prediction accuracy.

---

## 0. Honest diagnosis of v7.0

### What v7.0 does well

1. **Operator algebra predicts correctly.** Over 300 pipelines 100% accuracy.
   The engine knows before sampling which invariants a pipeline will have.

2. **Meta-theorems work.** 4/6 empirically strongly confirmed, 1 actively falsified.
   The system formulates universal statements and breaks them itself.

3. **Fixed-point characterization reveals structure.**
   Factorization shows that 3² × 11 and digit_sum ≡ 0 (mod 9) are dominant.
   Kaprekar constant 495 appears in non-Kaprekar pipelines.

### What v7.0 does NOT do (and thinks it does)

1. **"Symbolic reasoning" is still testing.**
   The meta-theorems are pre-defined templates that are empirically tested.
   The system does not *generate* theorems from data. It *verifies* handmade candidates.

2. **Fixed-point "solving" is constraint-search.**
   There is no algebraic derivation. `f(n) = n` is not solved —
   it is searched in a limited domain. That is fast brute-force, not algebra.

3. **Emergent mechanisms are co-occurrence labels.**
   "entropy_reducing + strong_convergence" is a statistical fact, not a causal model.
   The system cannot explain *why* entropy reduction causes convergence.

4. **No proof sketches.**
   "Strong empirical" is the endpoint. There is no attempt at even a
   proof direction. The engine could say: "Proof via well-ordering of ℕ
   and boundedness of output range" — but it doesn't.

5. **No cross-attractor structural analysis.**
   183 fixed points characterized, but nowhere an analysis of:
   - Why is digit_sum = 9 or 18 so dominant?
   - Why is 3² × 11 a recurring pattern in factorizations?
   - Are there universal digit_sum constraints on fixed points?

6. **No theory graph.**
   v5.0 had a theory graph. v7.0 lost it. There is no structure that
   connects theorems, categories, mechanisms and fixed points.

7. **Meta-theorems are not generated from data.**
   They are manually defined. A truly symbolic system would:
   - Detect patterns in fixed-point properties
   - Automatically make universal statements from them
   - Then test those

---

## 1. The fundamental architecture leap for v8.0

### From: "I test pre-defined theorems"
### To: "I derive theorems from structural patterns and suggest proof directions"

This requires four new capabilities:

---

## 2. Four new modules

### MODULE A: Proof Sketch Generator

**What it does:**
Given a confirmed meta-theorem, generate a proof skeleton.

**How:**
```
INPUT:  MetaTheorem(antecedent={MONOTONE, BOUNDED}, consequent="convergence")
OUTPUT: ProofSketch(
    strategy: "well_ordering",
    steps: [
        "1. Define sequence s_k = f^k(n) for arbitrary n in domain",
        "2. By MONOTONE: s_{k+1} < s_k for all k (strictly decreasing)",
        "3. By BOUNDED: s_k ≥ 0 for all k (bounded below)",
        "4. By well-ordering of ℕ: strictly decreasing bounded sequence terminates",
        "5. Therefore ∃K: s_K = s_{K+1}, i.e. f(s_K) = s_K  □"
    ],
    assumptions: ["MONOTONE means f(n) < n for ALL n > fixed_point, not just sampled"],
    gaps: ["Need to verify MONOTONE holds universally, not just empirically"]
)
```

**Proof strategy library:**
| Strategy | When | Template |
|-----------|---------|----------|
| well_ordering | MONOTONE + BOUNDED | Decreasing sequence in ℕ is finite |
| pigeonhole | BOUNDED + finite_range | Finitely many outputs → repetition |
| modular_arithmetic | PRESERVES_MOD_K | Residue class preservation through composition |
| contraction_mapping | CONTRACTIVE | Banach fixed-point (discrete version) |
| entropy_argument | ENTROPY_REDUCING + BOUNDED | Entropy decreasing in finite space → minimum |
| induction_on_digits | LENGTH_REDUCING | Induction on number of digits |

**Crucial:** Each ProofSketch contains `gaps[]` — what still needs to be proven.
This makes it honest. The system does not claim a proof, it suggests a direction.

---

### MODULE B: Inductive Theorem Generator

**What it does:**
Generate meta-theorems FROM data, not from templates.

**How:**

Step 1: Analyze all confirmed conjectures
```python
confirmed_conjectures = [c for c in all_conjectures if c.status == EMPIRICAL]
```

Step 2: Extract property patterns
```python
# Which combinations of operator properties lead to convergence?
for conj in confirmed:
    pipeline_props = algebra.predict(conj.pipeline)
    attractor_props = characterize(conj.attractor)
    
    # Record: {input_properties} → {output_property}
    implications.append(pipeline_props → attractor_props)
```

Step 3: Generalize
```python
# If 15/15 pipelines with {ENTROPY, BOUNDED} converge to
# an attractor with digit_sum ∈ {9, 18, 27}:
# → Generate theorem: "ENTROPY + BOUNDED → attractor.digit_sum ≡ 0 (mod 9)"
```

Step 4: Falsify
```python
# Actively search for pipelines with {ENTROPY, BOUNDED} where
# attractor.digit_sum ≢ 0 (mod 9)
```

**This is the core of v8.0:** theorems that the system itself invents.

---

### MODULE C: Fixed-Point Structural Analyzer

**What it does:**
Analyze ALL found fixed points as a set and discover universal patterns.

**v7.0 problem:** 183 fixed points individually characterized, but never analyzed as a group.

**Concrete analyses:**

1. **Digit-sum distribution over all fixed points**
   ```
   Expected output:
   digit_sum = 9:   47 fixed points (25.7%)
   digit_sum = 18:  38 fixed points (20.8%)
   digit_sum = 1:   22 fixed points (12.0%)
   digit_sum = 27:  15 fixed points (8.2%)
   → HYPOTHESIS: digit_sum of fixed points is almost always ≡ 0 (mod 9)
   ```

2. **Factorization patterns**
   ```
   Factor 3 present:  142/183 (77.6%)
   Factor 11 present:  58/183 (31.7%)
   Factor 3² × 11:      34/183 (18.6%)
   → HYPOTHESIS: Fixed points of digit-operation pipelines contain
     disproportionately often factor 3 and 11
   ```

3. **Palindrome analysis**
   ```
   Palindrome fixed points: 41/183 (22.4%)
   → Compare with base-rate palindromes in [1, 100000]: ~0.3%
   → HYPOTHESIS: Fixed points are ~75x more often palindrome than expected
   ```

4. **Cross-pipeline fixed-point overlap**
   ```
   Fixed point 0:  appears in 89% of pipelines (trivial)
   Fixed point 1:  appears in 34% of pipelines
   Fixed point 9:  appears in 12% of pipelines
   Fixed point 81: appears in 8% of pipelines
   → HYPOTHESIS: There exists a universal fixed-point hierarchy
     {0} ⊂ {0,1} ⊂ {0,1,9} ⊂ {0,1,9,81} that holds for
     every pipeline with digit_sum as component
   ```

**This is what a mathematician would do:** not report 183 individual facts,
but investigate the structure of the set itself.

---

### MODULE D: Theory Graph (Reinstallation + Upgrade)

**What it does:**
Connects all discovered objects in a directed graph.

**Node types:**
- `Operator` — individual digit operation with algebraic profile
- `Pipeline` — composition of operators
- `Invariant` — proven/empirical property
- `FixedPoint` — characterized fixed point
- `Mechanism` — emergent discovered mechanism
- `Theorem` — universal statement (pre-defined or induced)
- `ProofSketch` — proof direction for a theorem
- `Category` — conceptual class of pipelines

**Relation types:**
- `COMPOSES` — Operator → Pipeline
- `SATISFIES` — Pipeline → Invariant
- `CONVERGES_TO` — Pipeline → FixedPoint
- `EXPLAINED_BY` — Pipeline → Mechanism
- `SUPPORTS` — Pipeline → Theorem
- `FALSIFIES` — Pipeline → Theorem
- `PROVES_VIA` — Theorem → ProofSketch
- `MEMBER_OF` — Pipeline → Category
- `IMPLIES` — Invariant → Invariant (e.g., MONOTONE + BOUNDED → CONVERGENT)
- `SHARES_STRUCTURE` — FixedPoint → FixedPoint (same factorization pattern)

**Query capabilities:**
```python
# "Which theorems are supported by pipelines with digit_sum?"
graph.query(operator="digit_sum").theorems()

# "Which fixed points share factor 3² × 11?"
graph.query(factor_pattern={3: 2, 11: 1}).fixed_points()

# "Which proof sketches still have open gaps?"
graph.query(type="ProofSketch").filter(has_gaps=True)

# "Which categories are isomorphic?"
graph.query(type="Category").isomorphisms()
```

---

## 3. New output mode: Structured Discovery Report

v7.0 prints results to console. v8.0 must generate a structured report:

```markdown
# Discovery Report — Session 2026-02-24

## Universal Laws Discovered
### Law 1: Digit-Sum Mod-9 Preservation
- **Statement:** ∀P containing digit_sum: P preserves n mod 9
- **Status:** STRONG EMPIRICAL (300/300 pipelines)
- **Proof sketch:** By definition, digit_sum(n) ≡ n (mod 9).
  Composition with mod-9-preserving operators maintains this.
- **Gaps:** Need to verify all operators in pipeline preserve mod 9.

## Fixed-Point Universals
### Universal 1: Digit-Sum Divisibility
- **Statement:** For 77.6% of non-trivial fixed points, 3 | FP
- **Explanation:** digit_sum maps to mod-9 residue classes.
  Fixed points must satisfy f(n) = n, constraining digit structure.

## New Conceptual Categories
### Category: "9-Absorbers"
- Pipelines that converge to fixed points with digit_sum = 9
- 47 members identified
- Structural basis: mod-9 preservation + monotone reduction

## Open Problems
1. Is digit_sum = 18 dominant in fixed points of length ≥ 3?
2. Why does 3² × 11 appear so frequently in factorizations?
3. Does there exist a pipeline without trivial fixed point 0?
```

---

## 4. Concrete implementation order

### Phase 1: Fixed-Point Structural Analyzer (MODULE C)
- Lowest complexity, highest information yield
- Analyze the 183 fixed points from v7.0 as a set
- Generate hypotheses about digit_sum distribution, factorization, palindromes

### Phase 2: Inductive Theorem Generator (MODULE B)
- Use output from Phase 1 as input
- Generate theorems from fixed-point patterns
- Actively test them

### Phase 3: Proof Sketch Generator (MODULE A)
- For each confirmed theorem, generate proof direction
- Implement 6 proof strategy templates
- Mark gaps honestly

### Phase 4: Theory Graph (MODULE D)
- Connect everything
- Make queryable
- Generate Structured Discovery Report

---

## 5. What v8.0 should NOT do

- **Not claim that it proves.** ProofSketch ≠ Proof. Always mark gaps.
- **Not add more operators.** 19 is enough. Deepen, don't broaden.
- **Not more pipelines per session.** 300 is enough data. Analyze deeper.
- **Not try GPU.** The bottleneck is reasoning, not speed.
- **Not try to write a paper.** That is for the human.
  The system delivers the discoveries, the human delivers the publication.

---

## 6. The real goal

v7.0 says: "This theorem is empirically strong."
v8.0 must say:

> "Theorem MT002 (mod-9 attractor constraint) is empirically confirmed
> over 300 pipelines. Proof direction: digit_sum(n) ≡ n (mod 9) by
> definition. All operators in the tested pipelines preserve mod 9
> (proven via operator algebra). Composition of mod-9-preserving
> functions is mod-9-preserving (proven, MT004). Thus attractor ≡ input
> (mod 9). QED-sketch, gap: universality of operator algebra profiles
> for inputs > 10⁵."

THAT is the difference between detection and deduction.

---

## 7. One sentence summary

**v8.0 = v7.0 + "and this is why it is true, and this is what I don't yet know"**

---

## 8. Technical constraints

- **Python 3.11**, no external dependencies beyond numpy/scipy
- **SQLite** for persistence
- **No GPU** — pure CPU-symbolic
- **Max ~2000 lines** — keep it readable
- **Build on v7.0** — import or copy, no complete rewrite
- **Keep v7.0 unchanged** — v8.0 is a new file

---

*This document is the architecture prompt for the next session.
Read it, understand it, build it.*
