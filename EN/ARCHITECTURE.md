# SYNTRIAD Digit-Pipeline Analysis Framework

**Systematic computational exploration of algebraic structure in digit-based dynamical systems.**

A computational research framework that enumerates digit-operation pipelines in arbitrary bases, catalogues fixed points, classifies families, and verifies conjectures through exhaustive computation and formal proof.

---

## Key Results

| Metric | Value |
|--------|-------|
| Knowledge Base | 79 facts (65 proven) |
| Invariants per fixed point | 16 |
| Analysis phases | 19 |
| Digit operations | 22 |
| Modules | 30 (A–Z + R11) |
| Complement-closed FPs found | ~90 |
| Infinite FP families | **4 proven** (symmetric, 1089×m, sort_desc, palindromes) |
| Formal proofs verified | **12/12** |
| Armstrong numbers catalogued | k=1..7, k_max(10)=60 proven |
| Kaprekar constants | 3d (495), 4d (6174), 6d (549945, 631764) |
| digit_sum Lyapunov | Conditionally proven (DS061) |
| Multi-base support | b ∈ {5..16} |
| Unit tests | 117/117 passing |
| Runtime (200 pipelines) | ~58 seconds |

### The Central Discovery

Fixed points of digit-operation pipelines are governed by the algebraic structure of base `b`:

```
b ≡  1 (mod b-1)  → digit_sum ≡ n (mod b-1) → factor (b-1) enrichment
b ≡ -1 (mod b+1)  → alt_digit_sum ≡ n (mod b+1) → factor (b+1) enrichment
Base 10: (3 × 11)² = 1089 → universal fixed point at resonance intersection
```

Four proven infinite families of digit-operation fixed points:

1. **Symmetric family**: `d_i + d_{2k+1-i} = b-1` — exactly **(b-2)×b^(k-1)** FPs (**DS034**)
2. **1089×m family**: `(b-1)(b+1)²×m` for m=1..b-1 — **UNIVERSAL** (**DS040**)
3. **sort_desc family**: non-increasing digits — **C(k+9,k)-1** FPs (**DS062**)
4. **Palindrome family**: reverse-invariant — **9×10^(floor((k-1)/2))** FPs (**DS063**)

---

## Engine Evolution

```
v1.0   GPU Attractor Verification       Exhaustive brute-force verification (CUDA)
v2.0   GPU Rigorous Analysis            Methodological improvements, state-space bounding
v4.0   Discovery Engine v4               Pipeline enumeration + pattern detection
v5.0   Meta-Symmetry Engine             Operator embeddings, meta-learning, theory graph
v6.0   Invariant Discovery Engine       Structural abstraction, conjecture generation
v7.0   Symbolic Dynamics Engine         Operator algebra, FP solver, meta-theorem generator
v8.0   Deductive Theory Engine          Proof sketches, inductive theorems, theory graph
v9.0   Reasoning Engine v9               Knowledge base, pattern chains, anomaly detection
v10.0  Symbolic Dynamics Engine v10     Multi-base engine, algebraic FP classifier, Lyapunov search
v11.0  Formal Proof Engine              Computational proof verification, 5/5 formal proofs
v12.0  Complete Proof Engine             12/12 proofs, DS040 corrected, Lyapunov bounds, odd-length
v13.0  Broadened Discovery Engine        Armstrong/narcissistic, Kaprekar odd-base, orbit analysis
v14.0  Deep Analysis Engine              Universal Lyapunov, repunits, cycle taxonomy, multi-digit Kaprekar
v15.0  Open Questions Engine             4 FP families, Kaprekar d>3, Lyapunov proof, Armstrong bounds
```

The current engine (**v15.0 / R11**) integrates 11 rounds of analysis (R1–R11: DeepSeek R1-R5 + Manus R6 + Cascade R7-R11).

---

## Architecture (v14.0)

```
LAYER 1: Empirical Dynamics          Attractor detection, sampling, orbit analysis
LAYER 2: Operator Algebra + KB       Symbolic prediction, 71 facts (DS011–DS060)
LAYER 3: Symbolic Reasoning           FP solver, meta-theorems, proof sketches
LAYER 4: Deductive Theory             Induced theorems, theory graph
LAYER 5: Heuristic Pattern Detection   Pattern chains, anomaly detection, follow-up generation
LAYER 6: Multi-base Generalization     BaseNDigitOps, cross-base comparison
META:    Homeostatic self-regulation
```

### Modules (A–U)

| Module | Name | Session | Purpose |
|--------|------|---------|---------|
| A | DigitOps | v7.0 | 22 digit operations (reverse, complement, sort, pow, Kaprekar, gcd, xor, narcissistic) |
| B | OperatorAlgebra | v7.0 | Symbolic convergence prediction before sampling |
| C | FixedPointSolver | v7.0 | Constraint-based FP search + 16-invariant characterization |
| D | PipelineExplorer | v7.0 | Stochastic pipeline generation with adaptive scoring |
| E | KnowledgeBase | v9.0 | 51 proven/empirical facts, gap closure loop |
| F | CausalChainConstructor | v9.0 | Mechanistic explanations from statistical patterns |
| G | SurpriseDetector | v9.0 | Anomaly detection, follow-up question generation |
| H | GapClosureLoop | v9.0 | Proven facts automatically close proof gaps |
| I | SelfQuestioner | v9.0 | After each discovery: "why?" and "what follows?" |
| J | MonotoneAnalyzer | v9.0 | Decreasing measure detection for convergence proofs |
| K | BoundednessAnalyzer | v9.0 | Growth/reduction classification of pipelines |
| L | ComplementClosedFamilyAnalyzer | v9.0 | Multiset complement-closure, symmetric/1089 classification |
| M | MultiplicativeFamilyDiscovery | v9.0 | Multiplicative, reverse, and complement relations |
| **N** | **MultiBaseAnalyzer** | **R6** | BaseNDigitOps for arbitrary base b, cross-base FP analysis |
| **O** | **SymbolicFPClassifier** | **R6** | Algebraic FP conditions per pipeline (10 known conditions) |
| **P** | **LyapunovSearch** | **R6** | Grid-search for decreasing Lyapunov functions |
| **Q** | **FamilyProof1089** | **R6** | Algebraic proof of 1089×m complement-closure |
| **R** | **FormalProofEngine** | **R7–R8** | Computational verification of algebraic proofs (12/12) |
| **S** | **NarcissisticAnalyzer** | **R9** | Armstrong numbers catalog, bifurcation analysis per k |
| **T** | **OddBaseKaprekarAnalyzer** | **R9** | Kaprekar dynamics in odd bases: cycles vs fixed points |
| **U** | **OrbitAnalyzer** | **R9** | Convergence time, cycle detection per pipeline |
| **V** | **ExtendedPipelineAnalyzer** | **R10** | Long pipelines (5+ ops), FP saturation analysis |
| **W** | **UniversalLyapunovSearch** | **R10** | Universal Lyapunov function search (9 candidates) |
| **X** | **RepunitAnalyzer** | **R10** | Repunit connection to CC-families |
| **Y** | **CycleTaxonomy** | **R10** | Full attractor cycle classification per pipeline |
| **Z** | **MultiDigitKaprekar** | **R10** | Kaprekar dynamics for 4, 5, 6-digit numbers |

### 16 Analysis Phases

| Phase | Name | Session |
|-------|------|---------|
| 1 | Pipeline Exploration | v7.0 |
| 2 | Fixed-Point Structural Analysis | v7.0 |
| 3 | Causal Chain Construction | v9.0 |
| 4 | Surprise Detection | v9.0 |
| 5 | Gap Closure | v9.0 |
| 6 | Self-Questioning | v9.0 |
| 7 | Monotone Analysis | v9.0 |
| 8 | Boundedness Analysis | v9.0 |
| 9 | Complement-Closed Family Analysis | v9.0 |
| 10 | Multiplicative Family Discovery | v9.0 |
| 11 | Pipeline-Specific FP Classification | R5 |
| 12 | Multi-Base Engine | R6 |
| 13 | Algebraic FP-Characterization | R6 |
| 14 | Lyapunov Searcher | R6 |
| 15 | 1089-Family Algebraic Proof | R6 |
| **16** | **Formal Proof Verification** | **R7** |
| **17** | **Path B — Broader** | **R9** |
| **18** | **Path D — Deeper²** | **R10** |

---

## Knowledge Base (71 facts: DS011–DS060)

### Core theorems (DS011–DS023, R1–R5)

| ID | Statement | Level |
|----|-----------|-------|
| DS011 | Complement-closed ⇒ even digit count | PROVEN |
| DS012 | Complement-closed digit_sum = 9k | PROVEN |
| DS013 | All CC FPs divisible by 9 | PROVEN |
| DS014 | 5 complement pairs in base 10 | AXIOM |
| DS017 | Every 2-digit ds=9 number is FP of rev∘comp | PROVEN |
| DS018 | Complete 2-digit FP set: {18,...,81} (90 excluded) | PROVEN |
| DS019 | Digit multiset invariant under permutation ops | PROVEN |
| DS020 | Infinite family: 8×10^(k-1) FPs per 2k digits | PROVEN |
| DS023 | Pipelines without growth ops are bounded | PROVEN |

### R6 theorems (DS024–DS033, Manus)

| ID | Statement | Level |
|----|-----------|-------|
| DS024 | 1089×m complement-closed: digits = [m, m-1, 9-m, 10-m] | PROVEN |
| DS025 | Digit formula for 1089×m verified | PROVEN |
| DS026 | Multi-base symmetric FP formula: (b-2)×b^(k-1) | PROVEN |
| DS027 | Complement-closed digit_sum = k×(b-1) in base b | PROVEN |
| DS028 | Resonance factors (b-1) and (b+1) dominant | PROVEN |
| DS029 | Kaprekar constant for 3-digit is 495 (not 1089) | PROVEN |
| DS030 | reverse FPs = palindromes | PROVEN |
| DS031 | sort_desc∘sort_asc FPs = descending digits | PROVEN |
| DS032 | Lyapunov: digit_sum decreasing for n≥10 | PROVEN |
| DS033 | Multi-base formula verified for b∈{8,10,12,16} | EMPIRICAL |

### R7 theorems (DS034–DS040, Cascade)

| ID | Statement | Level |
|----|-----------|-------|
| DS034 | **FORMAL PROOF**: (b-2)×b^(k-1) for ALL bases b≥3 | PROVEN |
| DS035 | CC numbers divisible by (b-1) in any base | PROVEN |
| DS036 | comp∘comp = identity (d_1 ≤ b-2 only) | PROVEN |
| DS037 | rev∘rev = identity (no trailing zeros) | PROVEN |
| DS038 | digit_pow2(n) < n for n≥1000 (Lyapunov) | PROVEN |
| DS039 | Kaprekar K_b = (b/2)(b²-1) algebraically proven for even b≥4 | **PROVEN** |
| DS040 | 1089-family is **UNIVERSAL** for all bases b≥3 (corrected R8) | **PROVEN** |

### R8 theorems (DS041–DS045, Cascade)

| ID | Statement | Level |
|----|-----------|-------|
| DS041 | Odd-length rev∘comp has NO FPs in even bases | PROVEN |
| DS042 | Lyapunov: digit_pow3(n) < n for n≥10000 | PROVEN |
| DS043 | Lyapunov: digit_pow4(n) < n for n≥100000 | PROVEN |
| DS044 | Lyapunov: digit_pow5(n) < n for n≥1000000 | PROVEN |
| DS045 | Lyapunov: digit_factorial_sum(n) < n for n≥10000000 | PROVEN |

### R9 theorems (DS046–DS052, Cascade)

| ID | Statement | Level |
|----|-----------|-------|
| DS046 | Armstrong numbers per k are FINITE (Lyapunov argument) | PROVEN |
| DS047 | Armstrong k=3: exactly {153, 370, 371, 407} | PROVEN |
| DS048 | Armstrong k=4: exactly {1634, 8208, 9474} | PROVEN |
| DS049 | Even bases: Kaprekar 3-digit FP K_b unique | PROVEN |
| DS050 | Odd bases: Kaprekar 3-digit has FPs and/or cycles | EMPIRICAL |
| DS051 | New operations: digit_gcd, digit_xor, narcissistic_step (22 total) | AXIOM |
| DS052 | Odd-length rev∘comp FPs DO exist in odd bases | PROVEN |

### R10 theorems (DS053–DS060, Cascade)

| ID | Statement | Level |
|----|-----------|-------|
| DS053 | Long pipelines (5+ ops) yield no NEW FPs vs short pipelines | EMPIRICAL |
| DS054 | digit_sum is best universal Lyapunov candidate | EMPIRICAL |
| DS055 | Repunits R_k are NEVER complement-closed FPs | PROVEN |
| DS056 | (b-1)×R_k is always palindrome, never CC FP | PROVEN |
| DS057 | Kaprekar 4-digit constant = 6174, convergence ≤7 steps | PROVEN |
| DS058 | Kaprekar 5-digit: no unique FP, cycles and multiple FPs | EMPIRICAL |
| DS059 | Convergent pipelines have on average 1-3 unique attractors | EMPIRICAL |
| DS060 | Kaprekar 4-digit in base 8: 1656; base 12: 8286 | EMPIRICAL |

---

## Bugfixes

### v11→v12

- **DS040 formula**: was `(b-1)²(b+1)` (=891 for b=10), corrected to `(b-1)(b+1)²` (=1089)
- **DS040 claim**: was "unique to base 10", corrected to "**universal** for all bases b≥3"
- **DS039**: upgraded from EMPIRICAL to PROVEN with algebraic proof

### v10→v11

- **DS020 `six_digit_predicted`**: was `9×10²=900`, corrected to `8×10²=800` (d_1=9 gives leading zero)
- **sort_desc∘sort_asc condition**: condition text and formal spec were swapped with sort_asc∘sort_desc
- **DS036 complement involution**: added leading-zero exception (d_1=9 breaks involution)

---

## Repository Structure

```
.
├── pipeline_dsl.py          # M0: Pipeline DSL, operations, domain policies
├── experiment_runner.py     # M1: Experiment execution + SQLite storage
├── feature_extractor.py     # M2: Number profiling + conjecture mining
├── proof_engine.py          # M3: Proof skeletons + density estimation + ranking
├── appendix_emitter.py      # M4: Deterministic artifact generation + bundling
├── conftest.py              # Pytest configuration
│
├── scripts/
│   ├── run_experiments.py   # Recreate results.db from scratch (C1 audit fix)
│   └── reproduce.py         # One-command artifact generation + verification
│
├── tests/                   # 377+ pytest tests (M0–M4)
│   ├── test_m0.py
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_engine.py       # Legacy engine tests (98 unittest tests)
│
├── papers/                  # Manuscripts
│   ├── paper_A.tex          # Paper A: Algebraic structure + 5 infinite families
│   ├── paper_B.tex          # Paper B: Attractor spectra + epsilon-universality
│   ├── paper.tex            # DEPRECATED combined version
│   └── paper_draft.md       # DEPRECATED markdown draft
│
├── engines/                 # Research prototypes (NOT submission code)
│   ├── README.md            # Explains relationship to M0–M4
│   └── *.py                 # v4–v10 engines (historical)
│
├── docs/                    # Documentation
│   ├── ROADMAP_SUBMISSION.md      # Active submission preparation plan
│   ├── SYNTRIAD_ENGINE_vNext_AUDIT_REPORT.md
│   ├── roadmap.md                 # Research trajectory (R1–R11)
│   └── ...
│
└── data/                    # Data files
    ├── results.db           # Experiment results (SQLite)
    └── *.json               # Export files
```

### Submission Codebase (M0–M4)

| Module | File | Purpose |
|--------|------|---------|
| **M0** | `pipeline_dsl.py` | Pipeline DSL, frozen dataclasses, canonical hashing |
| **M1** | `experiment_runner.py` | Exhaustive/sampled experiments, SQLite storage |
| **M2** | `feature_extractor.py` | 17-feature number profiling, typed conjecture mining |
| **M3** | `proof_engine.py` | Proof skeletons, density estimation, heuristic ranking |
| **M4** | `appendix_emitter.py` | LaTeX appendices, manifests, determinism guard, bundling |

### Research Prototypes (engines/)

Historical engines (v4–v10) preserved for reference. See `engines/README.md`.

---

## Quick Start

```bash
# Step 1: Run experiments (recreates results.db from scratch)
python scripts/run_experiments.py

# Step 2: Generate and verify artifacts
python scripts/reproduce.py --db data/results.db --bundle

# Run all tests
python -m pytest tests/ -q
```

**Requirements:** Python 3.10+, NumPy. No other dependencies.

---

## Feedback Rounds

| Round | Agent | Key Additions |
|-------|-------|---------------|
| **R1** | DeepSeek | alt_digit_sum, digital_root, is_niven |
| **R2** | DeepSeek | cross_sum, hamming_weight, MonotoneAnalyzer, BoundednessAnalyzer |
| **R3** | DeepSeek | KnowledgeBase (DS011–DS016), ComplementClosedFamilyAnalyzer |
| **R4** | DeepSeek | DS017–DS019, canonical pipelines, multiset bug fix |
| **R5** | DeepSeek | DS020–DS023, infinite family theorem, leading-zero correction |
| **R6** | Manus | Modules N–Q, DS024–DS033, multi-base engine, 1089 proof |
| **R7** | Cascade | Module R, DS034–DS040, formal proofs 5/5, bugfixes, 47 unit tests |
| **R8** | Cascade | DS041–DS045, DS039/040 upgraded, 12/12 proofs, DS040 corrected, 57 tests |
| **R9** | Cascade | Path B: Modules S–U, DS046–DS052, 22 ops, Armstrong, odd-base Kaprekar, orbits, 76 tests |
| **R10** | Cascade | Path D: Modules V–Z, DS053–DS060, Lyapunov, repunits, cycle taxonomy, multi-digit Kaprekar, 98 tests |

### Iterative Corrections

- **R5**: Computation revealed `9×10^(k-1)` → `8×10^(k-1)` (leading-zero exclusion)
- **R7**: Fixed sort_desc∘sort_asc condition swap, DS036 leading-zero exception, DS034 verification method
- **R8**: Discovered DS040 formula error: `(b-1)²(b+1)` → `(b-1)(b+1)²`, proving 1089-family is universal

---

## Hardware

- **GPU:** NVIDIA RTX 4000 Ada (used for v1/v2 exhaustive verification)
- **CPU:** 32-core Intel i9
- **RAM:** 64 GB
- **OS:** Windows

---

## Next Steps

See [`docs/ROADMAP_SUBMISSION.md`](docs/ROADMAP_SUBMISSION.md) for submission preparation plan,
and [`docs/roadmap.md`](docs/roadmap.md) for research trajectory history (R1–R11).

---

## License

Part of the SYNTRIAD research portfolio. Internal use.
