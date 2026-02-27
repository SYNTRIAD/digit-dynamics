# SYNTRIAD Digit-Dynamics Discovery Engine — English Documentation

**Complete technical reference for the digit-dynamics research project.**

[← Back to root](../README.md) | [Nederlandse versie →](../NL/README.md)

---

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Research Engines (v1–v15)](#research-engines-v1v15)
- [Reproducibility Infrastructure (M0–M4.1)](#reproducibility-infrastructure-m0m41)
- [Running the Code](#running-the-code)
- [Test Suite](#test-suite)
- [Papers](#papers)
- [Mathematical Results](#mathematical-results)
- [Knowledge Base](#knowledge-base)
- [Research Process](#research-process)
- [Reproducing Results](#reproducing-results)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project investigates fixed-point structure in composed digit-operation dynamical systems. Given a pipeline of digit operations (reverse, complement, sort, digit-sum, Kaprekar step, 1089-trick, etc.) applied iteratively to natural numbers, the system identifies which numbers are fixed points, classifies them algebraically, and proves structural results that hold across all number bases b ≥ 3.

The codebase has two tracks:
1. **Research engines** (v1–v15): Exploratory, single-file engines used for discovery and conjecture generation.
2. **M0–M4.1 modules**: Modular, deterministic infrastructure for reproducible results and paper appendix generation.

---

## Installation

### Requirements
- Python 3.10+
- NumPy (only external dependency for core engines)
- pytest (for test suite)

### Setup

```bash
cd EN
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Optional
- CUDA toolkit + RTX GPU (for GPU-accelerated exhaustive verification in v1–v2 engines)
- LaTeX distribution with amsart (for compiling papers)

---

## Directory Structure

```
EN/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT license
│
├── engines/                   # Research engines (exploration track)
│   ├── gpu_attractor_verification.py    # v1.0 — GPU brute-force
│   ├── gpu_rigorous_analysis.py         # v2.0 — Methodological refinement
│   ├── autonomous_discovery_engine_v4.py # v4.0 — Self-generating pipelines
│   ├── meta_symmetry_engine_v5.py       # v5.0 — Operator embeddings
│   ├── invariant_discovery_engine_v6.py # v6.0 — Structural abstraction
│   ├── symbolic_dynamics_engine_v7.py   # v7.0 — Operator algebra
│   ├── deductive_theory_engine_v8.py    # v8.0 — Proof sketches
│   └── research_engine_v15.py           # v15.0 — Current (30 modules)
│
├── src/                       # Reproducibility infrastructure (M0–M4.1)
│   ├── pipeline_dsl.py        # M0: Canonical semantics & hashing
│   ├── experiment_runner.py   # M1: SQLite store & batch runner
│   ├── feature_extractor.py   # M2: Feature extraction & conjecture mining
│   ├── proof_engine.py        # M3: Proof skeletons & ranking model
│   ├── appendix_emitter.py    # M4: LaTeX appendix & manifest generation
│   ├── reproduce.py           # M4.1: One-command reproducibility runner
│   └── conftest.py            # Pytest configuration
│
├── tests/                     # Test suite
│   ├── test_m0.py             # M0 tests: parsing, hashing, operations
│   ├── test_m1.py             # M1 tests: store, batch runner, export
│   ├── test_m2.py             # M2 tests: features, orbits, conjectures
│   ├── test_m3.py             # M3 tests: skeletons, density, ranking
│   ├── test_m4.py             # M4 tests: manifest, LaTeX, determinism
│   └── test_engine.py         # Legacy tests for research engines
│
├── papers/                    # LaTeX manuscripts
│   ├── paper_A.tex            # Paper A: Pure mathematics (9 theorems)
│   ├── paper_A.pdf            # Compiled Paper A
│   ├── paper_B.tex            # Paper B: Experimental mathematics
│   ├── paper_B.pdf            # Compiled Paper B
│   └── paper.tex              # Combined working draft
│
├── data/                      # Generated data & databases
│   ├── results.db             # SQLite experiment database
│   ├── results_export.json    # JSON export of all experiments
│   └── paper_b_hashes.json    # Verification hashes for Paper B
│
└── docs/                      # Extended documentation
    ├── WHAT_WE_DISCOVERED.md
    ├── EVOLUTION_FROM_SCRIPTS_TO_REASONING.md
    ├── PUBLICATION_STRATEGY.md
    ├── FORMAL_VERIFICATION_REPORT.md
    └── REFLECTION_V8.md / REFLECTION_V10.md
```

---

## Research Engines (v1–v15)

The research engines form the *discovery track* — single-file scripts that evolved progressively:

| Version | Engine | Lines | Key Capability |
|---------|--------|-------|----------------|
| v1.0 | gpu_attractor_verification | ~500 | CUDA exhaustive verification |
| v2.0 | gpu_rigorous_analysis | ~550 | State-space bounding, cycle detection |
| v4.0 | autonomous_discovery_engine | ~900 | Self-generating pipeline exploration |
| v5.0 | meta_symmetry_engine | ~1,300 | Operator embeddings, meta-learning |
| v6.0 | invariant_discovery_engine | ~1,500 | Structural abstraction, conjecture generation |
| v7.0 | symbolic_dynamics_engine | ~1,400 | Operator algebra, 100% symbolic prediction |
| v8.0 | deductive_theory_engine | ~1,800 | Proof sketches, induced theorems |
| v15.0 | research_engine_v15 | ~6,500 | 30 modules, 6 reasoning layers, 83 KB facts |

**To run the current research engine:**

```bash
python engines/research_engine_v15.py
```

This runs a full session: initializes 22 digit operations, loads the knowledge base (83 facts), runs multi-base analysis, and outputs results to stdout.

**Note:** The research engines use `random` for stochastic pipeline generation. Results vary between runs. For deterministic, reproducible results, use the M0–M4 infrastructure.

---

## Reproducibility Infrastructure (M0–M4.1)

The M0–M4.1 modules form the *formalization track* — designed for deterministic, hash-verified reproducibility:

### M0: Canonical Semantics (pipeline_dsl.py)

The foundation. Provides:
- **Layer A (Semantic)**: `OperationSpec`, `Pipeline`, `DomainPolicy` — frozen dataclasses, pure data
- **Layer B (Execution)**: `OperationExecutor`, `PipelineRunner` — implementations, strictly separated from Layer A

Key design: Pipeline identity is determined by canonical JSON → SHA-256, never by string representation. Whitespace, separator choice (`|>`, `->`, `>>`), and formatting are irrelevant.

```python
from pipeline_dsl import OperationRegistry, Pipeline, DomainPolicy, PipelineRunner

reg = OperationRegistry()  # 22 operations with full metadata
pipe = Pipeline.parse("kaprekar_step |> digit_pow4 |> digit_sum", registry=reg)
domain = DomainPolicy.paper_a_kaprekar(k=4)

runner = PipelineRunner(reg)
result = runner.run_exhaustive(pipe, domain)
print(f"Fixed points: {result.fixed_points}")
print(f"Result hash: {result.sha256}")
```

### M1: Experiment Runner (experiment_runner.py)

SQLite-backed experiment store with:
- Versioned schema (`SCHEMA_VERSION = "1.0"`)
- Batch runner for multiple pipelines × domains
- JSON export with full hash chains

### M2: Feature Extractor (feature_extractor.py)

Per-number, per-orbit, and per-pipeline feature extraction:
- 17 number features (digit_sum, palindrome, sortedness, entropy, ...)
- Orbit analysis (contraction ratio, transient length, cycle detection)
- 6 conjecture types: COUNTING, MODULAR, MONOTONICITY, UNIVERSALITY, STRUCTURE, INVARIANT
- Delta-debugging falsification engine

### M3: Proof Engine (proof_engine.py)

Structural reasoning layer:
- **Proof Skeleton Generator**: Identifies proof strategy (MOD_INVARIANT, BOUNDING, COUNTING_RECURRENCE, ...), reduction steps, and remaining gaps
- **Counterexample Density Estimator**: Clopper-Pearson bounds, Rule of Three, calibrated confidence
- **Pattern Compressor**: Detects affine, polynomial, modular, and recurrence patterns
- **Conjecture Mutator**: Generalization, transfer, strengthening, weakening
- **Ranking Model v1.0**: Explicit weights (empirical 0.30, structural 0.25, novelty 0.20, simplicity 0.15, falsifiability 0.10), versioned and logged

### M4: Appendix Emitter (appendix_emitter.py)

Generates review-proof artifacts:
- Paper A / Paper B appendix LaTeX (domain-separated)
- Canonical JSON manifests and catalogs
- DeterminismGuard: rerun verification with byte-identical checking
- ArtifactPackager: zip bundle with README, environment snapshot, lockfile

### M4.1: Reproducibility Runner (reproduce.py)

One-command orchestration:

```bash
python src/reproduce.py --db data/results.db --out repro_out --bundle
```

---

## Running the Code

### Quick verification

```bash
# Run all unit + integration tests (~7 seconds)
pytest tests/ -v -m "not exhaustive"

# Run only M0 tests (fastest, ~1 second)
pytest tests/test_m0.py -v

# Run exhaustive tests (20+ minutes, all k-ranges)
pytest tests/ -v -m exhaustive
```

### Generate paper appendices

```bash
python src/reproduce.py --db data/results.db --out repro_out --bundle
```

This produces:
- `repro_out/appendix_paper_a.tex`
- `repro_out/appendix_paper_b.tex`
- `repro_out/repro_manifest.json`
- `repro_out/reproducibility_bundle.zip`

### Run a specific experiment

```python
from src.pipeline_dsl import OperationRegistry, Pipeline, DomainPolicy, PipelineRunner

reg = OperationRegistry()
pipe = Pipeline.parse("truc_1089 |> digit_pow4", registry=reg)
domain = DomainPolicy(base=10, digit_length=4, exclude_repdigits=True)
runner = PipelineRunner(reg)
result = runner.run_exhaustive(pipe, domain)

print(f"Attractors: {result.num_attractors}")
print(f"Fixed points: {result.fixed_points}")
print(f"Convergence rate: {result.convergence_rate:.6f}")
print(f"Basin entropy: {result.basin_entropy:.6f}")
print(f"SHA-256: {result.sha256}")
```

---

## Test Suite

| Suite | File | Tests | Coverage | Runtime |
|-------|------|-------|----------|---------|
| M0: Canonicalization | test_m0.py | ~50 | Parsing, hashing, operations, golden freezes | ~1s |
| M1: Experiment Store | test_m1.py | ~30 | Store, batch runner, JSON export | ~2s |
| M2: Feature Extraction | test_m2.py | ~30 | Number features, orbits, conjectures | ~2s |
| M3: Proof Engine | test_m3.py | ~50 | Skeletons, density, patterns, ranking | ~2s |
| M4: Appendix Emitter | test_m4.py | ~80 | Manifest, LaTeX, determinism, integration | ~3s |
| Legacy: Research Engines | test_engine.py | ~98 | Operations, KB facts, multi-base, proofs | ~10s |
| **Total** | | **~338** | | **~20s** |

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Papers

### Paper A: "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases"

Pure mathematics paper. Contains 9 theorems with algebraic proofs:

1. **Symmetric FP count**: rev∘comp produces (b−2)·b^(k−1) fixed points among 2k-digit numbers
2. **Universal 1089-family**: A_b = (b−1)(b+1)² for all bases b ≥ 3
3. **Four infinite families**: Explicit counting formulas, pairwise disjoint
4. **Fifth family**: 1089-trick fixed points n_k = 110·(10^(k−3)−1) for k ≥ 5
5. **Kaprekar constants**: K_b = (b/2)(b²−1) for even bases
6. **Armstrong upper bound**: k_max(b) ≤ ⌊b·log(b)/log(b−1)⌋ + 1
7. **Repunit exclusion**: Repunit multiples are not fixed points of rev∘comp

Target: Journal of Integer Sequences or Integers.

### Paper B: "Attractor Spectra and ε-Universality in Digit-Operation Dynamical Systems"

Experimental mathematics paper. Introduces:
- **ε-universality**: Quantitative measure of attractor dominance
- **Basin entropy**: Shannon entropy of basin-fraction distribution
- **Composition lemma**: ε-bound for composed pipelines
- **Conditional Lyapunov theorem**: Convergence guarantees for operations in class P ∪ C
- **Three conjectures** (C1: basin entropy monotonicity, C2: asymptotic ε-universality, C3: attractor count growth)

Target: Experimental Mathematics.

### Compiling

```bash
cd papers
pdflatex paper_A.tex
pdflatex paper_B.tex
```

---

## Mathematical Results

### The 22 Digit Operations

| Operation | Notation | Class | ds-class |
|-----------|----------|-------|----------|
| reverse | rev(n) | Permutation | P |
| complement_9 | comp(n) | Digitwise map | P |
| digit_sum | ds(n) | Aggregate | C |
| digit_product | dp(n) | Aggregate | C |
| digit_pow2–5 | dp_k(n) | Aggregate | X |
| sort_asc / sort_desc | sort↑/↓(n) | Permutation | P |
| kaprekar_step | kap(n) | Subtractive | C |
| truc_1089 | T(n) | Mixed | C |
| add_reverse | n + rev(n) | Mixed | X |
| sub_reverse | |n − rev(n)| | Mixed | C |
| swap_ends | swap(n) | Permutation | P |
| rotate_left/right | rot(n) | Permutation | P |
| digit_factorial_sum | dfs(n) | Aggregate | X |
| digit_gcd | dgcd(n) | Aggregate | C |
| digit_xor | dxor(n) | Aggregate | C |
| collatz | col(n) | Arithmetic | X |

Classes: P = digit-sum preserving, C = contractive, X = expansive.

### The Core Algebraic Insight

Fixed points of digit-operation pipelines are determined by the algebraic structure of the number base:
- **10 ≡ 1 (mod 9)** → digit_sum preserves residue mod 9 → factor-3 enrichment in fixed points
- **10 ≡ −1 (mod 11)** → alternating digit-sum structure → factor-11 enrichment
- **(3 × 11)² = 1089** → the universal resonance point where both structures intersect
- Generalizes: **A_b = (b−1)(b+1)²** for every base b ≥ 3

### Knowledge Base (83 Facts)

The research engine maintains a knowledge base of 83 facts (DS011–DS072):
- 72 proven (algebraic or exhaustive proof)
- 11 conjectured (strong empirical evidence)

Spanning: complement-closed families, symmetric FP counts, Kaprekar constants, 1089-universality, Lyapunov descent bounds, Armstrong bounds, repunit exclusion, orbit analysis.

---

## Research Process

### Multi-Agent Collaboration

| Round | Agent | Focus |
|-------|-------|-------|
| R1–R5 | DeepSeek | Mathematical consultation, conjecture refinement |
| R6 | Manus | Multi-base engine, bulk implementation |
| R7–R8 | Claude/Cascade | Formal proof verification (12/12) |
| R9 | Claude/Cascade | Armstrong, Kaprekar odd-base, orbit analysis |
| R10 | Claude/Cascade | Universal Lyapunov, repunits, cycle taxonomy |
| R11 | Claude/Cascade | Open questions, fifth family, publication prep |

Human researcher (R. Havenaar) directed all phases, identified algebraic structures, and made conceptual connections.

### Self-Correction Examples

The system's epistemological health is demonstrated by autonomous corrections:
- **R5**: Engine detected that DeepSeek's prediction (9×10^(k−1) FPs) was wrong → corrected to 8×10^(k−1) (leading-zero exclusion)
- **R8**: Engine discovered formula error in DS040: (b−1)²(b+1) → (b−1)(b+1)², verified algebraically
- **v7.0**: Engine falsified its own meta-theorem "Monotone+Bounded → convergence" with concrete counterexample

---

## Reproducing Results

### Full reproducibility pipeline

```bash
# 1. Set deterministic hash seed (recommended)
export PYTHONHASHSEED=0

# 2. Run reproducibility runner
python src/reproduce.py --db data/results.db --out repro_out --bundle

# 3. Verify output
# The final line prints FINAL MANIFEST SHA256
# This should match the hash published in the paper appendix
```

### What reproduce.py does

1. Checks runtime determinism knobs (PYTHONHASHSEED)
2. Prints environment summary
3. Generates `requirements.lock.txt` (pip freeze)
4. Runs M4 emitter: DB → manifest + catalogs + LaTeX appendices
5. Runs DeterminismGuard: reruns in temp directory, compares byte-for-byte
6. Packages `reproducibility_bundle.zip`
7. Prints final manifest SHA-256

### Hash chain

```
OperationRegistry.sha256
    └── Pipeline.sha256 (canonical JSON of op sequence)
        └── DomainPolicy.sha256 (base, digit_length, exclusions)
            └── RunResult.sha256 (all numeric results, fixed precision)
                └── Manifest.sha256 (all of the above + environment)
```

Every link in the chain is deterministic within the same Python version and platform.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pipeline_dsl'`**
→ Make sure you're running from the `EN/src/` directory, or add it to `PYTHONPATH`:
```bash
export PYTHONPATH=EN/src:$PYTHONPATH
```

**`FileNotFoundError: results.db`**
→ The experiment database must exist before running `reproduce.py`. It is included in the repository under `data/results.db`.

**Different manifest hash on rerun**
→ Check: (1) Same Python version? (2) Same NumPy version? (3) PYTHONHASHSEED=0? Float formatting at the 12th decimal digit may differ across platforms.

**LaTeX compilation errors**
→ Paper A requires `amsart` document class. Install a full TeX distribution (TeX Live or MikTeX).

---

*SYNTRIAD Research — February 2026*
