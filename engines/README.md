# engines/ — Research Prototypes (Not Part of Submission)

This directory contains the historical research engine prototypes (v1–v10)
that were used during the iterative discovery process (R1–R11). They are
preserved for **reference and reproducibility of the research trajectory**
but are **not part of the submission codebase**.

## Submission Codebase (M0–M4)

The submission-quality code lives in the project root:

| Module | File | Purpose |
|--------|------|---------|
| M0 | `pipeline_dsl.py` | Pipeline DSL, operations, domain policies |
| M1 | `experiment_runner.py` | Experiment execution + SQLite storage |
| M2 | `feature_extractor.py` | Number profiling + conjecture mining |
| M3 | `proof_engine.py` | Proof skeletons + density estimation + ranking |
| M4 | `appendix_emitter.py` | Deterministic artifact generation + bundling |

## Contents of this Directory

| File | Version | Description |
|------|---------|-------------|
| `autonomous_discovery_engine_v4.py` | v4.0 | Pipeline enumeration prototype |
| `meta_symmetry_engine_v5.py` | v5.0 | Operator embeddings, meta-learning |
| `invariant_discovery_engine_v6.py` | v6.0 | Structural abstraction, conjecture generation |
| `symbolic_dynamics_engine_v7.py` | v7.0 | Operator algebra, FP solver |
| `deductive_theory_engine_v8.py` | v8.0 | Proof sketches, inductive theorems |
| `abductive_reasoning_engine_v9.py` | v9.0 | Knowledge base, pattern chains |
| `abductive_reasoning_engine_v10.py` | v10–v15 | Full 26-module research engine (6,500 lines) |

## Relationship to M0–M4

The M0–M4 modules were **extracted and refactored** from the monolithic v10
engine into a clean, modular architecture with:
- Frozen dataclasses for semantic specifications
- Separated execution logic
- Canonical hashing for reproducibility
- Deterministic artifact generation

The engines in this directory import from M0 (`pipeline_dsl.py`) for shared
data types but are otherwise self-contained.
