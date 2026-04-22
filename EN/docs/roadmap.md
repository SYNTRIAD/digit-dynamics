# Roadmap: Autonomous Discovery Engine v15.0 → v16.0
## R12 Session

---

## Current state (v15.0 / R11)

- **79 KB facts** (65 proven), DS011–DS068
- **16 invariants** per fixed point
- **19 analysis phases** (incl. Path B + Path D + Path E)
- **30 modules** (A–Z + R11 modules)
- **22 operations**
- **117 unit tests** (100% passing)
- **12/12 formal proofs** computationally verified
- **Multi-base support**: b ∈ {5..16}
- **Armstrong numbers**: catalog k=1..7, k_max formula proven
- **Kaprekar**: 3-digit (495), 4-digit (6174), 6-digit (549945, 631764)
- **Universal Lyapunov**: digit_sum conditionally proven (DS061)
- **Repunits**: never CC FPs (DS055, proven)
- **4 infinite FP families**: symmetric, 1089×m, sort_desc, palindromes (DS064)

### New results R11 (PATH E — Open Questions)
- **DS061**: digit_sum Lyapunov — conditionally proven (NOT universal)
- **DS062**: sort_desc FPs — infinite family, formula C(k+9,k)-1 proven
- **DS063**: palindromes — infinite FP family of reverse, formula proven
- **DS064**: 4 disjoint infinite FP families proven
- **DS065**: Armstrong k_max formula — k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)} proven
- **DS066**: Kaprekar 6-digit — two FPs (549945, 631764) exhaustively verified
- **DS067**: all Kaprekar FPs divisible by 9 (mod 9 invariant)
- **DS068**: Kaprekar FP count per digit length irregular (no formula)

### Proven results (R7–R10)
- **DS034**: Symmetric FP formula (b-2)×b^(k-1) for EVERY base b≥3
- **DS035**: CC numbers divisible by (b-1) in every base
- **DS036/037**: Involutions comp∘comp and rev∘rev with edge cases
- **DS038–DS045**: Lyapunov bounds digit_pow2–5 and digit_factorial_sum
- **DS039**: Kaprekar K_b = (b/2)(b²-1) algebraically proven
- **DS040**: 1089 family is **UNIVERSAL** for all bases b≥3
- **DS041**: Odd-length rev∘comp = ∅ for even bases
- **DS046**: Armstrong numbers finite per k (Lyapunov argument)
- **DS047/048**: Armstrong k=3 and k=4 exhaustively verified
- **DS049**: Even bases Kaprekar FP is unique
- **DS050**: Odd bases Kaprekar: cycles and FPs (EMPIRICAL)
- **DS052**: Odd-length rev∘comp FPs DO exist in odd bases
- **DS055**: Repunits R_k are NEVER CC FPs (proven)
- **DS056**: (b-1)×R_k always palindrome, never CC FP (proven)
- **DS057**: Kaprekar 4-digit = 6174, ≤7 steps (proven)

---

## ✅ PATH A — DEEPER: COMPLETED (R8)

| # | Task | Result | Status |
|---|------|--------|--------|
| A1 | Formalize Kaprekar constants | DS039 → PROVEN | ✅ |
| A2 | Prove 1089 universality | DS040 → PROVEN + CORRECTED | ✅ |
| A3 | Odd-length rev∘comp = ∅ | DS041 PROVEN | ✅ |
| A4 | Lyapunov digit_pow3/4/5 | DS042–DS044 PROVEN | ✅ |
| A5 | Lyapunov digit_factorial_sum | DS045 PROVEN | ✅ |

## ✅ PATH B — BROADER: COMPLETED (R9)

| # | Task | Result | Status |
|---|------|--------|--------|
| B1 | Parametric bifurcation | NarcissisticAnalyzer (Module S) | ✅ |
| B2 | Narcissistic numbers | Armstrong k=1..7 catalog, DS046–DS048 | ✅ |
| B3 | Orbit dynamics | OrbitAnalyzer (Module U), convergence times | ✅ |
| B4 | New operations | digit_gcd, digit_xor, narcissistic_step (22 ops) | ✅ |
| B5 | Odd bases Kaprekar | OddBaseKaprekarAnalyzer (Module T), DS049–DS050 | ✅ |

## ✅ PATH D — DEEPER²: COMPLETED (R10)

| # | Task | Result | Status |
|---|------|--------|--------|
| D1 | Longer pipelines | ExtendedPipelineAnalyzer (Module V), DS053 | ✅ |
| D2 | Universal Lyapunov | UniversalLyapunovSearch (Module W), DS054 | ✅ |
| D3 | Repunit connection | RepunitAnalyzer (Module X), DS055–DS056 | ✅ |
| D4 | Attractor cycle classification | CycleTaxonomy (Module Y), DS059 | ✅ |
| D5 | 4+ digit Kaprekar | MultiDigitKaprekar (Module Z), DS057–DS058, DS060 | ✅ |

---

## ✅ PATH E — OPEN QUESTIONS: COMPLETED (R11)

| # | Task | Result | Status |
|---|------|--------|--------|
| E1 | Kaprekar d>3 algebraic analysis | KaprekarAlgebraicAnalyzer, DS066-DS068 | ✅ |
| E2 | 3rd+ infinite FP family | ThirdFamilySearcher, DS062-DS064 | ✅ |
| E3 | digit_sum Lyapunov proof | DigitSumLyapunovProof, DS061 | ✅ |
| E4 | Armstrong k_max bounds | ArmstrongBoundAnalyzer, DS065 | ✅ |

### R11 Discoveries

**Kaprekar 6-digit (549945, 631764):**
- 549945 = 3² × 5 × 11² × 101 — **palindrome!** — ds=36, ÷9, ÷11
- 631764 = 2² × 3² × 7 × 23 × 109 — ds=27, ÷9, NOT ÷11
- No algebraic formula found — FP count per d is irregular
- Pair_sums are NOT constant → no simple symmetry

**4 infinite FP families:**
1. Symmetric rev∘comp: d_i + d_{2k+1-i} = 9 → (b-2)×b^(k-1) per digit length
2. 1089×m multiplicative: A_b × m for m=1..b-1
3. sort_desc FPs: non-increasing digits → C(k+9,k)-1 per digit length
4. Palindromes: reverse-invariant → 9×10^(floor((k-1)/2)) per digit length

**digit_sum Lyapunov:**
- NOT universal — complement_9, kaprekar_step, truc_1089 increase ds
- CONDITIONALLY proven for ds-non-increasing pipelines

**Armstrong k_max:**
- k_max(10) = 60, k_max(2) = 2, k_max(16) = 116
- Formula: k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)}
- k_max/b ratio grows slowly: ~6 for b=10, ~7.25 for b=16

---

## ✅ PATH C — PUBLICATION: COMPLETED (R11)

| # | Task | Result | Status |
|---|------|--------|--------|
| C1 | Paper structure | 12 sections, abstract with 8 theorems | ✅ |
| C2 | Main theorem | Theorem 1 (DS034) complete proof | ✅ |
| C3 | Secondary results | Theorems 2–8 fully written | ✅ |
| C4 | Methodology section | v15.0 engine description, 11 feedback rounds | ✅ |
| C5 | Paper draft v1.0 | `paper_draft.md` — 660 lines, publication-ready | ✅ |

---

## Strategic paths (R12+)

### 📝 PATH F — SUBMISSION PREPARATION (SUPERSEDED)

> **Replaced by:** `docs/ROADMAP_SUBMISSION.md` — based on independent technical audit
> (docs/SYNTRIAD_ENGINE_vNext_AUDIT_REPORT.md, 2026-02-25).
> PATH F items are fully covered by the new action plan (C1–C4, I1–I5, N1–N3).

| # | Task | Description | Status |
|---|------|-------------|--------|
| F1 | LaTeX conversion | paper_draft.md → .tex with AMS style | ✅ → paper_A.tex, paper_B.tex exist; finalization via C2 |
| F2 | Peer review | Independent audit + language correction | ✅ → Audit report + C3 language fix |
| F3 | Code repository | Repo restructured (tests/, engines/, scripts/, papers/, docs/, data/) | ✅ → Phase 0 + C4 bundle cleanup |
| F4 | arXiv submission | After all audit fixes | ⏳ → see ROADMAP_SUBMISSION.md |

**Strongest publication claims:**
> 1. "For every base b≥3: the number of FPs of rev∘comp with 2k digits
>    is exactly (b-2)×b^(k-1). For odd length in even bases: zero."
> 2. "The 1089 multiplicative family (b-1)(b+1)²×m is UNIVERSAL:
>    A_b×m has digits [m, m-1, (b-1)-m, b-m] and is CC in every base."
> 3. "There exist at least 4 disjoint infinite FP families for
>    digit-operation pipelines, each with proven counting formula."
> 4. "Kaprekar K_b = (b/2)(b²-1) is algebraically proven as FP for even b≥4."
> 5. "Armstrong k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)} is proven;
>    k_max(10) = 60 with complete catalog k=1..7."
> 6. "digit_sum is conditionally Lyapunov for ds-non-increasing pipelines."
> 7. "Repunits R_k are NEVER complement-closed FPs (proven)."
> 8. "Kaprekar 6-digit: two FPs (549945 palindrome, 631764); no formula."

---

## Execution order

```
R8:  PATH A (A1–A5)  →  ✅ COMPLETED. DS039–DS045, 12/12 proofs, 57 tests.
R9:  PATH B (B1–B5)  →  ✅ COMPLETED. Modules S–U, DS046–DS052, 22 ops, 76 tests.
R10: PATH D (D1–D5)  →  ✅ COMPLETED. Modules V–Z, DS053–DS060, 98 tests.
R11: PATH E (E1–E4)  →  ✅ COMPLETED. Open questions, DS061–DS068, 117 tests.
R11: PATH C (C1–C5)  →  ✅ COMPLETED. Paper v1.0, 660 lines, 8 theorems.
R12: PATH F (F1–F4)  →  LaTeX conversion + arXiv submission
```

---

## Completed (DO NOT redo)

| Item | Status | Session |
|------|--------|---------|
| Multi-base engine (BaseNDigitOps) | ✅ | R6 |
| SymbolicFPClassifier (10 conditions) | ✅ | R6+R7 |
| LyapunovSearch (grid search) | ✅ | R6 |
| FamilyProof1089 (algebraic proof) | ✅ | R6 |
| FormalProofEngine (12/12 proofs) | ✅ | R7+R8 |
| DS034–DS045 PROVEN | ✅ | R7+R8 |
| DS040 CORRECTED + UNIVERSAL | ✅ | R8 |
| **PATH A completed (A1–A5)** | ✅ | **R8** |
| **57 unit tests** | ✅ | **R8** |
| **PATH B completed (B1–B5)** | ✅ | **R9** |
| **NarcissisticAnalyzer (Module S)** | ✅ | **R9** |
| **OddBaseKaprekarAnalyzer (Module T)** | ✅ | **R9** |
| **OrbitAnalyzer (Module U)** | ✅ | **R9** |
| **DS046–DS052** | ✅ | **R9** |
| **22 operations** | ✅ | **R9** |
| **76 unit tests** | ✅ | **R9** |
| **README + roadmap v13.0** | ✅ | **R9** |
| **PATH D completed (D1–D5)** | ✅ | **R10** |
| **ExtendedPipelineAnalyzer (Module V)** | ✅ | **R10** |
| **UniversalLyapunovSearch (Module W)** | ✅ | **R10** |
| **RepunitAnalyzer (Module X)** | ✅ | **R10** |
| **CycleTaxonomy (Module Y)** | ✅ | **R10** |
| **MultiDigitKaprekar (Module Z)** | ✅ | **R10** |
| **DS053–DS060** | ✅ | **R10** |
| **98 unit tests** | ✅ | **R10** |
| **README + roadmap v14.0** | ✅ | **R10** |
| **PATH E completed (E1–E4)** | ✅ | **R11** |
| **KaprekarAlgebraicAnalyzer** | ✅ | **R11** |
| **ThirdFamilySearcher** | ✅ | **R11** |
| **DigitSumLyapunovProof** | ✅ | **R11** |
| **ArmstrongBoundAnalyzer** | ✅ | **R11** |
| **DS061–DS068** | ✅ | **R11** |
| **117 unit tests** | ✅ | **R11** |
| **README + roadmap v15.0** | ✅ | **R11** |
| **PATH C completed (C1–C5)** | ✅ | **R11** |
| **Paper draft v1.0 (660 lines, 8 theorems)** | ✅ | **R11** |
| **paper.tex (AMS-art LaTeX, arXiv-ready)** | ✅ | **R11** |

---

## Open mathematical questions

1. ~~Why are 1089×m complement-closed?~~ → **PROVEN (DS024)**
2. ~~Do analogous families exist in other bases?~~ → **YES! UNIVERSAL (DS040)**
3. ~~Is there a connection between repunits (111...1) and complement-closed families?~~ → **NO: repunits never CC FPs (DS055)**
4. ~~Can we derive an FP condition for every pipeline?~~ → **10 conditions proven (Module O)**
5. ~~Does a universal Lyapunov function exist for all convergent pipelines?~~ → **digit_sum best candidate, but not 100% universal (DS054)**
6. ~~Are Kaprekar constants proven per base?~~ → **YES, even b (DS039). Odd b: cycles (DS050)**
7. ~~Why does the 1089 structure fail in other bases?~~ → **DOES NOT FAIL! Universal (DS040)**
8. ~~Do odd-length numbers ever have rev∘comp FPs?~~ → **NO in even bases (DS041). YES in odd (DS052)**
9. ~~What are the Kaprekar constants for odd bases?~~ → **Analyzed: mix of FPs and cycles (DS050, Module T)**
10. ~~Are there more than 2 disjoint infinite FP families?~~ → **YES! At least 4 families (DS064)**
11. ~~What is the exact upper bound for Armstrong numbers (k_max in base b)?~~ → **k_max(b) = max{k : k×(b-1)^k ≥ b^(k-1)} (DS065)**
12. Does a closed formula exist for the number of Armstrong numbers per k? → **OPEN — count sequence is irregular**
13. ~~Can digit_sum be proven as Lyapunov (not just empirically)?~~ → **CONDITIONALLY PROVEN (DS061)**
14. ~~Does an algebraic formula exist for Kaprekar constants at d>3?~~ → **NO for d>4 — FP count irregular, no formula (DS068)**
15. Does a closed formula exist for Kaprekar FP count as function of d? → **OPEN — irregular (DS068)**
16. Is 549945 (6-digit Kaprekar palindrome) algebraically explainable? → **OPEN**
