# SYNTRIAD Digit Dynamics Discovery Engine

**From brute-force to reasoning in three days.**

An autonomous mathematical research system that discovers, classifies, and proves properties of digit-operation dynamical systems. Starting from GPU brute-force computation, the system evolved through six layers of reasoning — from empirical observation to abductive explanation — in 72 hours.

---

## The Discovery

When you compose digit operations (reverse, digit-sum, sort, complement, ...) and iterate them on natural numbers, the resulting dynamical systems have fixed points and attractors that are **not random**. They are deeply rooted in the algebraic structure of the number base itself:

- `10 ≡ 1 (mod 9)` → digit-sum invariance → factor-3 enrichment
- `10 ≡ -1 (mod 11)` → alternating structure → factor-11 enrichment
- `(3 × 11)² = 1089` → universal fixed point at the resonance crossing
- This generalizes to **all bases** `b ≥ 3` via `(b-1)` and `(b+1)`

**Key results:** 9 theorems, 5 infinite fixed-point families, 83 knowledge-base facts, 12 computationally verified proofs.

---

## The System

The engine evolved from v1.0 (GPU calculator) to v15.0 (30-module abductive reasoning system) in three days (23–26 February 2026), orchestrated by a human researcher with three AI systems (DeepSeek, Manus, Claude/Cascade).

### Six Layers of Reasoning

```
Layer 6 ──── Multi-base Generalization ──── "Does this hold in EVERY base?"
Layer 5 ──── Abductive Reasoning ────────── "WHY is this true?"
Layer 4 ──── Deductive Theory ───────────── "What FOLLOWS from this?"
Layer 3 ──── Symbolic Reasoning ─────────── "What do I PREDICT?"
Layer 2 ──── Operator Algebra + KB ──────── "What do I KNOW for certain?"
Layer 1 ──── Empirical Dynamics ─────────── "What do I SEE?"
Meta   ──── Homeostatic Self-Regulation ── "Am I functioning well?"
```

### Self-Steering Cycle

After each research session, the system generates:
1. **REFLECTION** — honest self-analysis: what is real, what is noise?
2. **SELF_PROMPT** — concrete instructions for the next version

This created a feedback loop that drove the evolution from computation to comprehension.

---

## Repository Structure

```
├── README.md                  ← You are here
├── ARCHITECTURE.md            ← Detailed technical architecture (M0-M4)
├── requirements.txt           ← Python dependencies
│
├── src/                       ← Core submission code (M0-M4 modules)
│   ├── pipeline_dsl.py        ← M0: Canonical semantics & reproducibility
│   ├── experiment_runner.py   ← M1: Experiment execution + SQLite store
│   ├── feature_extractor.py   ← M2: Feature extraction + conjecture mining
│   ├── proof_engine.py        ← M3: Proof skeletons + density estimation
│   └── appendix_emitter.py    ← M4: Deterministic artifact generation
│
├── engines/                   ← Historical research prototypes (v4-v15)
│   ├── README.md
│   ├── autonomous_discovery_engine_v4.py   ← Exploration + hypothesis
│   ├── meta_symmetry_engine_v5.py          ← Meta-learning + theory graph
│   ├── invariant_discovery_engine_v6.py    ← Structural abstraction
│   ├── symbolic_dynamics_engine_v7.py      ← Operator algebra + FP solver
│   ├── deductive_theory_engine_v8.py       ← Proof sketches + induction
│   ├── abductive_reasoning_engine_v9.py    ← Causal chains + self-questioning
│   └── abductive_reasoning_engine_v10.py   ← Full 30-module system (v15)
│
├── early_scripts/             ← Phase I: GPU brute-force & iterative search
│   ├── symmetry_discovery_engine.py   ← v1.0: First engine, 22 operations
│   ├── gpu_symmetry_hunter.py         ← CUDA kernels, 150M samples/sec
│   ├── gpu_deep_researcher.py         ← GPU + self-adaptation
│   ├── scoring_engine_v2.py           ← Triviality filter
│   ├── meta_discovery_engine.py       ← Self-improving search
│   ├── autonomous_researcher.py       ← Dynamic operation generation
│   ├── extended_research_session.py   ← v4: Long-running autonomous
│   ├── extended_research_session_v5.py ← v5: Cycles + genetic mutation
│   ├── extended_research_session_v6.py ← v6: ML predictor
│   └── ...                            ← Supporting scripts
│
├── scripts/                   ← Utility & verification scripts
│   ├── gpu_attractor_verification.py  ← Exhaustive GPU verification
│   ├── gpu_rigorous_analysis.py       ← State-space bounding
│   ├── reproduce.py                   ← Reproduction script
│   └── run_experiments.py             ← Experiment runner
│
├── tests/                     ← Test suite
│   ├── test_m0.py .. test_m4.py       ← M0-M4 module tests
│   ├── test_engine.py                 ← Engine integration tests
│   ├── test_operations.py             ← Operation correctness tests
│   ├── test_gpu_kernels.py            ← GPU kernel tests
│   └── test_regression.py             ← Regression tests
│
├── papers/                    ← LaTeX papers + compiled PDFs
│   ├── paper_A.tex / .pdf     ← Paper A: 9 theorems (pure math)
│   ├── paper_B.tex / .pdf     ← Paper B: Experimental dynamics
│   └── paper_b_hashes.json    ← Verification hashes
│
├── data/                      ← Research data & results
│   ├── results.db             ← SQLite results database
│   ├── results_export.json    ← Exported results (JSON)
│   └── attractor_verification_report.json
│
└── docs/                      ← Documentatie
    ├── EVOLUTIE_VAN_SCRIPTS_TOT_REDENEREN.md   ← ★ Hoofddocument: de evolutie
    ├── WAT_WE_ONTDEKTEN.md                     ← De P_k-ontdekking (toegankelijk)
    ├── DISCOVERY_ENGINES_DOCUMENTATION.md       ← Engine-versiedetails
    ├── claude-opus-isomorfie-redeneerpatroon.md ← Isomorfie-analyse
    ├── observaties_symbolic_dynamics_engine.md   ← Manus-observaties
    ├── ADVERSARIAL_AUDIT_REPORT.md              ← Adversarial audit
    ├── FORMAL_VERIFICATION_REPORT.md            ← Verificatieresultaten
    ├── PUBLICATION_STRATEGY.md                  ← A/B/C publicatiestrategie
    ├── MANUS_PROMPT_R6.md                       ← Manus-sessieprompt
    ├── LINKEDIN_REDENEREN_KUN_JE_LEREN.md       ← LinkedIn-post
    ├── beschouwingen/                           ← Beschouwende essays
    │   ├── BESCHOUWING_PROJECT.md
    │   └── BESCHOUWING_DRIEHOEKSDIALOOG_P_k.md
    ├── reflecties/                              ← Zelfsturende documenten
    │   ├── REFLECTION_V8.md
    │   ├── REFLECTION_V10.md
    │   ├── SELF_PROMPT_V8.md
    │   └── SELF_PROMPT_V10.md
    └── protocol-simulatie/                      ← Research 2.0 protocol (Manus)
        ├── OPERATIONAL SEMANTICS.md
        ├── RESEARCH 2.0_ DELIVERY SUMMARY.md
        ├── STRUCTURAL ANALYSIS REPORT.md
        └── C003_ REVISED STRUCTURAL ANALYSIS.md
```

---

## Quick Start

### Requirements

- Python 3.10+
- NumPy
- Optional: CUDA-capable GPU (for early_scripts GPU acceleration)

```bash
pip install -r requirements.txt
```

### Run the Discovery Engine

```bash
# Run the full v15 research session (~58 seconds)
python engines/abductive_reasoning_engine_v10.py

# Run M0-M4 experiment pipeline
python scripts/run_experiments.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## The Evolution (3 Days)

| Day | Phase | What happened |
|-----|-------|---------------|
| **Day 1** (Feb 23) | Compute → Explore | GPU brute-force, 150M samples/sec. Pattern detection. First hypotheses. |
| **Day 1-2** (Feb 23-24) | Explore → Understand | Operator algebra (100% prediction accuracy). Proof sketches. Theory graph. |
| **Day 2-3** (Feb 24-25) | Understand → Explain | Knowledge base (83 facts). Causal chains. Surprise detection. Self-questioning. |
| **Day 3-4** (Feb 25-26) | Explain → Justify | M0-M4 modular refactoring. Research 2.0 protocol. P_k discovery. Papers. |

Read the full story: [`docs/EVOLUTIE_VAN_SCRIPTS_TOT_REDENEREN.md`](docs/EVOLUTIE_VAN_SCRIPTS_TOT_REDENEREN.md)

---

## Multi-Agent Orchestration

Three AI systems contributed, each for their strengths:

- **DeepSeek** (R1-R5): Deep mathematical consultation
- **Manus** (R6): Bulk implementation & protocol execution
- **Claude/Cascade** (R7-R11): Formal proofs & architectural leaps

A human researcher orchestrated the cycle, chose the sequence, and made the conceptual leaps that none of the AI systems could make independently.

---

## Key Documents

| Document | Audience | Focus |
|----------|----------|-------|
| [`EVOLUTIE_VAN_SCRIPTS_TOT_REDENEREN.md`](docs/EVOLUTIE_VAN_SCRIPTS_TOT_REDENEREN.md) | Everyone | Complete evolution story |
| [`WAT_WE_ONTDEKTEN.md`](docs/WAT_WE_ONTDEKTEN.md) | Everyone | The P_k projection discovery |
| [`claude-opus-isomorfie-redeneerpatroon.md`](docs/claude-opus-isomorfie-redeneerpatroon.md) | Researchers | Isomorphism to agent architectures |
| [`ADVERSARIAL_AUDIT_REPORT.md`](docs/ADVERSARIAL_AUDIT_REPORT.md) | Reviewers | Adversarial audit of the papers |
| [`paper_A.pdf`](papers/paper_A.pdf) | Mathematicians | 9 theorems, 5 infinite families |
| [`paper_B.pdf`](papers/paper_B.pdf) | Experimentalists | Dynamics framework, 3 conjectures |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{syntriad2026digit,
  title={Algebraic Structure of Fixed Points in Composed Digit-Operation Dynamical Systems},
  author={Havenaar, Remco and SYNTRIAD Research},
  year={2026},
  note={Autonomous discovery engine with six reasoning layers},
  url={https://github.com/SYNTRIAD/digit-dynamics}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

*SYNTRIAD Research — February 2026*
*"Redeneren kun je leren."*
