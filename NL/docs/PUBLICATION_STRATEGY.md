# Publication Strategy & Engine vNext Architecture

## Status na R12

- **Engine**: v16.0, 83 KB-feiten (DS001–DS072, 69 bewezen), 117 tests, 22 operaties
- **Paper**: v3, 8 pagina's, 9 theorema's + 3 conjectures + appendix
- **Vijfde familie bewezen**: truc_1089 FPs, n_k = 110×(10^(k-3)−1)

---

## A. Publicatiestrategie: Paper A/B Split

### Paper A — Wiskunde (hard)

**Titel**: "Fixed points of digit-operation pipelines in arbitrary bases"

**Scope**:
- Theorem 1: rev∘comp symmetric FPs — (b−2)·b^(k−1)
- Theorem 2: Universal 1089-family
- Theorem 3: Four infinite families (disjointness + counting)
- Theorem 4: Fifth family (truc_1089 FPs)
- Theorem 5: Kaprekar constants (3d algebraic + 4d/6d exhaustive)
- Theorem 6: Armstrong upper bound k_max(b)
- Theorem 7: Repunit exclusion
- Appendix: verificatieprocedure (pseudocode, search space, hashing)

**Wat NIET in Paper A**:
- Attractor spectra / ε-universality
- Basin entropy
- Conjectures C1–C3
- Engine-beschrijving

**Target journals**: Journal of Integer Sequences, Integers, Fibonacci Quarterly

**Geschatte lengte**: 12–15 pagina's

### Paper B — Experimenteel/Dynamisch

**Titel**: "Attractor spectra and ε-universality in digit-operation dynamical systems"

**Scope**:
- Definitie ε-universality + basin entropy
- Composition lemma
- Conditional Lyapunov theorem (met formele operatieklassen P/C/X)
- Lyapunov descent bounds
- GPU-exhaustive attractor tabel (4+ pipelines)
- Conjectures C1–C3
- Dataset release

**Target journals**: Experimental Mathematics, Complex Systems

**Geschatte lengte**: 10–12 pagina's

### Paper C — AI Method (optioneel, later)

**Titel**: "Domain-specific conjecture mining in discrete dynamical systems"

**Scope**: Engine als methodologisch onderwerp, evaluatiemetrics, conjecture yield, falsification rate

**Target**: AI for Math workshops (ICML, NeurIPS), AITP

---

## B. Engine vNext Architectuur

### Huidige codebase → Module mapping

| Huidig bestand | Functie | vNext Module |
|----------------|---------|--------------|
| `abductive_reasoning_engine_v10.py` | Hoofdengine, 22 ops, KB, 30 modules | M0 (Pipeline DSL) + M1 (Runner) |
| `autonomous_discovery_engine_v4.py` | BasinAnalyzer, HypothesisGenerator | M1 + M3 (Conjecture Gen) |
| `gpu_attractor_verification.py` | CUDA kernels, exhaustieve verificatie | M1 (GPU backend) |
| `test_engine.py` | 117 unit tests | Blijft, uitbreiden |
| `open_questions_analysis.py` | Ad-hoc analyse Q1–Q4 | M2 (Feature Extractor) |
| `invariant_discovery_engine_v6.py` | Invariant search | M2 + M3 |
| `deductive_theory_engine_v8.py` | Deductieve bewijzen | M6 (Proof Assistant) |

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

**Prioriteit**: KRITISCH (zonder dit geen reproduceerbaarheid)

**Wat het doet**:
- Formele definitie elke operatie (leading zero policy, digit-length behavior)
- Pipeline = geordende tuple van operatie-IDs
- SHA-256 hash per pipeline (unieke identifier)
- Serialisatie naar JSON

**Implementatie**: Nieuw bestand `pipeline_dsl.py`
- `@dataclass PipelineSpec(ops: Tuple[str], base: int, digit_policy: str)`
- `def canonical_hash(spec) -> str`
- Alle 22 operaties met expliciete edge-case documentatie

### M1: Experiment Runner + Result Store

**Wat verandert**: SQLite/Parquet output i.p.v. ad-hoc prints

**Hergebruik**: `gpu_attractor_verification.py` (CUDA kernels), `BasinAnalyzer`

**Nieuw**:
- `results.db` met schema: pipeline_hash, domain, attractor_set, basin_fractions, avg_steps, witness_traces
- Deterministische runs met seed
- Sampling vs exhaustief: expliciet label

### M2: Feature Extractor

**Wat het doet**: Per getal en per orbit structurele features berekenen

**Features per getal**: ds(n), n mod (b-1), n mod (b+1), complement-closure score, palindroom flag, sortedness (Kendall τ)

**Features per pipeline**: operatieklasse-signatuur (P/C/X), empirische monotonicity, contraction ratio

**Hergebruik**: `OperatorAlgebra`, `MonotoneAnalyzer`, `ComplementClosedFamilyAnalyzer`

### M3: Conjecture Generator

**Mechanismen**:
1. Pattern mining over parameters (bases, k) → zoek gesloten vormen
2. Invariant discovery: "∀n: f(n) ≡ 0 mod 9"
3. Attractor-structuur: "unieke FP voor k=..."
4. Counting conjectures: "#{FPs} = C(k+a, b)"

**Output**: `Conjecture(quantifier, predicate, evidence, exceptions)`

**Hergebruik**: `HypothesisGenerator` uit v4, `SelfQuestioner`

### M4: Conjecture Ranker

**Scores**:
- novelty (niet tautologisch)
- simplicity (kortste formule)
- stability (cross-base robuustheid)
- surprise (hoog effect)
- proof_likelihood (algebraïsche hooks aanwezig?)

**Implementatie**: Gewogen score, heuristiek-gebaseerd (geen ML nodig)

### M5: Falsification & Refinement

**Hergebruik**: Exhaustieve verificatie uit engine, `FormalProofEngine`

**Nieuw**:
- Property-based testing (hypothesis → targeted counterexample search)
- Delta-debugging: minimaliseer counterexamples
- Automatische hypothese-verfijning (exclude repdigits, require k even, etc.)

### M6: Proof Assistant Hooks

**Hergebruik**: `FormalProofEngine`, `FamilyProof1089`, deductive engine

**Nieuw**:
- Detecteer of claim reduceert tot digit-pair equations
- Genereer lemma-kandidaten (mod invariants, bounds, closure)
- Exporteer bewijs-skeleton naar LaTeX

### M7: Artifact Generator

**Output**: LaTeX tabellen, dataset dumps met checksums, Methods-sectie tekst

---

## C. Roadmap in 3 milestones

### M1: Reproducibility Backbone (1–2 weken)

- [ ] `pipeline_dsl.py` met canonical specs + hashing
- [ ] Leading zero / digit-length policy documentatie
- [ ] Result store (SQLite) met schema
- [ ] Alle 22 operaties: edge-case tests toevoegen

### M2: Conjecture Mining MVP (2–3 weken)

- [ ] Feature extractor module
- [ ] Conjecture templates (counts, invariants, universality)
- [ ] Ranker v0 (heuristiek)
- [ ] Falsification loop met delta-debugging

### M3: Proof Skeleton + Paper Split (1–2 weken)

- [ ] Automatic reduction patterns
- [ ] Paper A/B LaTeX templates
- [ ] Dataset release prep (checksums, README)
- [ ] Submission-ready Paper A

---

## D. Eén harde aanbeveling

**Maak "digit-length policy" en "operation semantics" absoluut expliciet en versioned.**

De huidige engine heeft impliciete conventies:
- `reverse_digits`: `int(str(n)[::-1])` → leading zero drops → digit count kan dalen
- `complement`: `(10^k - 1) - n` → vereist dat k bekend is
- `kaprekar_step`: `sort_desc - sort_asc` → behoudt digit count? (hangt af van leading zeros)
- `truc_1089`: `abs(n - rev(n))` → digit count kan veranderen

Dit moet in M0 expliciet worden vastgelegd vóór verdere conjecture mining.
