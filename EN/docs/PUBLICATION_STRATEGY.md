# Publication Strategy & Engine vNext Architecture

## Status after R12

- **Engine**: v16.0, 83 KB facts (DS001–DS072, 69 proven), 117 tests, 22 operations
- **Paper**: v3, 8 pages, 9 theorems + 3 conjectures + appendix
- **Fifth family proven**: truc_1089 FPs, n_k = 110×(10^(k-3)−1)

---

## A. Publication Strategy: Paper A/B Split

### Paper A — Mathematics (hard)

**Title**: "Fixed points of digit-operation pipelines in arbitrary bases"

**Scope**:
- Theorem 1: rev∘comp symmetric FPs — (b−2)·b^(k−1)
- Theorem 2: Universal 1089-family
- Theorem 3: Four infinite families (disjointness + counting)
- Theorem 4: Fifth family (truc_1089 FPs)
- Theorem 5: Kaprekar constants (3d algebraic + 4d/6d exhaustive)
- Theorem 6: Armstrong upper bound k_max(b)
- Theorem 7: Repunit exclusion
- Appendix: verification procedure (pseudocode, search space, hashing)

**What NOT in Paper A**:
- Attractor spectra / ε-universality
- Basin entropy
- Conjectures C1–C3
- Engine description

**Target journals**: Journal of Integer Sequences, Integers, Fibonacci Quarterly

**Estimated length**: 12–15 pages

### Paper B — Experimental/Dynamic

**Title**: "Attractor spectra and ε-universality in digit-operation dynamical systems"

**Scope**:
- Definition of ε-universality + basin entropy
- Composition lemma
- Conditional Lyapunov theorem (with formal operation classes P/C/X)
- Lyapunov descent bounds
- GPU-exhaustive attractor table (4+ pipelines)
- Conjectures C1–C3
- Dataset release

**Target journals**: Experimental Mathematics, Complex Systems

**Estimated length**: 10–12 pages

### Paper C — AI Method (optional, later)

**Title**: "Domain-specific conjecture mining in discrete dynamical systems"

**Scope**: Engine as methodological subject, evaluation metrics, conjecture yield, falsification rate

**Target**: AI for Math workshops (ICML, NeurIPS), AITP

---

## B. Engine vNext Architecture

### Current codebase → Module mapping

| Current file | Function | vNext Module |
|----------------|---------|--------------|
| `abductive_reasoning_engine_v10.py` | Main engine, 22 ops, KB, 30 modules | M0 (Pipeline DSL) + M1 (Runner) |
| `autonomous_discovery_engine_v4.py` | BasinAnalyzer, HypothesisGenerator | M1 + M3 (Conjecture Gen) |
| `gpu_attractor_verification.py` | CUDA kernels, exhaustive verification | M1 (GPU backend) |
| `test_engine.py` | 117 unit tests | Stays, extend |
| `open_questions_analysis.py` | Ad-hoc analysis Q1–Q4 | M2 (Feature Extractor) |
| `invariant_discovery_engine_v6.py` | Invariant search | M2 + M3 |
| `deductive_theory_engine_v8.py` | Deductive proofs | M6 (Proof Assistant) |

### vNext Modules

```
┌─────────────────────────────────────────────────┐
│              AUTONOMY LOOP                       │
│                                                  │
│  M0: Pipeline DSL ──→ M1: Experiment Runner     │
│         │                    │                   │
│         ▼                    ▼                   │
│  M2: Feature Extractor ──→ M3: Conjecture Gen   │
│                              │                   │
│                              ▼                   │
│                    M4: Conjecture Ranker          │
│                              │                   │
│                    ┌─────────┴──────────┐        │
│                    ▼                    ▼        │
│          M5: Falsification      M6: Proof Assist │
│                    │                    │        │
│                    └─────────┬──────────┘        │
│                              ▼                   │
│                    M7: Artifact Generator         │
│                              │                   │
│                    ┌─────────┴──────────┐        │
│                    ▼                    ▼        │
│              Paper Tables         KB Update      │
└─────────────────────────────────────────────────┘
```

### M0: Canonical Pipeline DSL

**Priority**: CRITICAL (without this no reproducibility)

**What it does**:
- Formal definition of each operation (leading zero policy, digit-length behavior)
- Pipeline = ordered tuple of operation IDs
- SHA-256 hash per pipeline (unique identifier)
- Serialization to JSON

**Implementation**: New file `pipeline_dsl.py`
- `@dataclass PipelineSpec(ops: Tuple[str], base: int, digit_policy: str)`
- `def canonical_hash(spec) -> str`
- All 22 operations with explicit edge-case documentation

### M1: Experiment Runner + Result Store

**What changes**: SQLite/Parquet output instead of ad-hoc prints

**Reuse**: `gpu_attractor_verification.py` (CUDA kernels), `BasinAnalyzer`

**New**:
- `results.db` with schema: pipeline_hash, domain, attractor_set, basin_fractions, avg_steps, witness_traces
- Deterministic runs with seed
- Sampling vs exhaustive: explicit label

### M2: Feature Extractor

**What it does**: Compute structural features per number and per orbit

**Features per number**: ds(n), n mod (b-1), n mod (b+1), complement-closure score, palindrome flag, sortedness (Kendall τ)

**Features per pipeline**: operation class signature (P/C/X), empirical monotonicity, contraction ratio

**Reuse**: `OperatorAlgebra`, `MonotoneAnalyzer`, `ComplementClosedFamilyAnalyzer`

### M3: Conjecture Generator

**Mechanisms**:
1. Pattern mining over parameters (bases, k) → search for closed forms
2. Invariant discovery: "∀n: f(n) ≡ 0 mod 9"
3. Attractor structure: "unique FP for k=..."
4. Counting conjectures: "#{FPs} = C(k+a, b)"

**Output**: `Conjecture(quantifier, predicate, evidence, exceptions)`

**Reuse**: `HypothesisGenerator` from v4, `SelfQuestioner`

### M4: Conjecture Ranker

**Scores**:
- novelty (not tautological)
- simplicity (shortest formula)
- stability (cross-base robustness)
- surprise (high impact)
- proof_likelihood (algebraic hooks present?)

**Implementation**: Weighted score, heuristic-based (no ML needed)

### M5: Falsification & Refinement

**Reuse**: Exhaustive verification from engine, `FormalProofEngine`

**New**:
- Property-based testing (hypothesis → targeted counterexample search)
- Delta-debugging: minimize counterexamples
- Automatic hypothesis refinement (exclude repdigits, require k even, etc.)

### M6: Proof Assistant Hooks

**Reuse**: `FormalProofEngine`, `FamilyProof1089`, deductive engine

**New**:
- Detect if claim reduces to digit-pair equations
- Generate lemma candidates (mod invariants, bounds, closure)
- Export proof skeleton to LaTeX

### M7: Artifact Generator

**Output**: LaTeX tables, dataset dumps with checksums, Methods section text

---

## C. Roadmap in 3 milestones

### M1: Reproducibility Backbone (1–2 weeks)

- [ ] `pipeline_dsl.py` with canonical specs + hashing
- [ ] Leading zero / digit-length policy documentation
- [ ] Result store (SQLite) with schema
- [ ] All 22 operations: edge-case tests added

### M2: Conjecture Mining MVP (2–3 weeks)

- [ ] Feature extractor module
- [ ] Conjecture templates (counts, invariants, universality)
- [ ] Ranker v0 (heuristic)
- [ ] Falsification loop with delta-debugging

### M3: Proof Skeleton + Paper Split (1–2 weeks)

- [ ] Automatic reduction patterns
- [ ] Paper A/B LaTeX templates
- [ ] Dataset release prep (checksums, README)
- [ ] Submission-ready Paper A

---

## D. One hard recommendation

**Make "digit-length policy" and "operation semantics" absolutely explicit and versioned.**

The current engine has implicit conventions:
- `reverse_digits`: `int(str(n)[::-1])` → leading zero drops → digit count can decrease
- `complement`: `(10^k - 1) - n` → requires that k is known
- `kaprekar_step`: `sort_desc - sort_asc` → preserves digit count? (depends on leading zeros)
- `truc_1089`: `abs(n - rev(n))` → digit count can change

This must be explicitly established in M0 before further conjecture mining.
