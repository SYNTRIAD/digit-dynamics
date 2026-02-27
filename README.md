# SYNTRIAD Digit-Dynamics Discovery Engine

**Systematic computational exploration of fixed-point structure in digit-operation dynamical systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

A computational engine for exploring, classifying, and verifying algebraic structure in composed digit-operation systems. Evolved through 15 versions and 11 human-guided research sessions, combining multi-agent AI collaboration with algebraic reasoning to identify and computationally verify 9 theorems across arbitrary number bases (b ≥ 3).

![Convergence Pattern](assets/convergence-pattern.png)

---

## 🎯 Key Discovery: The P_k Projection

**The Problem:** When you iterate digit operations (reverse, digit-sum, sort) on numbers, they usually just shrink to single digits. Mathematically uninteresting.

**The Insight:** Add **fixed-length padding** (the P_k projection) and rich algebraic structure emerges:
- 9 theorems with algebraic proofs, verified exhaustively
- 5 infinite fixed-point families with explicit counting formulas
- Universal patterns across **all number bases** (b ≥ 3)
- Resonance structure determined by base arithmetic: (b−1)(b+1)² = 1089 in base 10

**The deeper point:** The fixed points are not accidents — they are forced by the algebraic structure of positional number systems. Specifically, 10 ≡ 1 (mod 9) and 10 ≡ −1 (mod 11) determine which numbers survive iterated digit operations.

Read more: [What We Discovered](EN/docs/WHAT_WE_DISCOVERED.md) | [Emergence Essay](assets/emergence-mechanisms.md)

---

## 📚 Repository Structure

This repository contains both English and Dutch versions:

### **[→ English Version (EN/)](EN/)**
Complete documentation, papers, code, and research artifacts in English.

### **[→ Nederlandse Versie (NL/)](NL/)**  
Volledige documentatie, papers, code en onderzoeksartefacten in het Nederlands.

### **[→ Assets](assets/)**
Visualizations and essays exploring universal patterns.

---

## 🚀 Quick Start

```bash
# English version
cd EN
pip install -r requirements.txt

# Run the research engine (v15, latest)
python engines/research_engine_v15.py

# Run reproducibility pipeline (M0-M4 modules)
python src/reproduce.py --db results.db --out repro_out --bundle

# Run all tests
pytest tests/ -v
```

See [EN/README.md](EN/README.md) for full documentation.

---

## 📄 Publications

### Papers (arXiv pending)
- **Paper A:** "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases"  
  Pure mathematics — 9 theorems, 5 infinite families, multi-base generalization
  
- **Paper B:** "Attractor Spectra and ε-Universality in Digit-Operation Dynamical Systems"  
  Experimental mathematics — novel ε-universality framework, exhaustive verification over 10⁷ inputs

### Candidate OEIS Sequence
- **a(n) = 110×(10^(n+1) − 1)** for n ≥ 1  
  Fixed points of the 1089-trick map for (n+4)-digit numbers

---

## 🧬 The Evolution

The engine evolved through 15 versions across 11 feedback rounds, guided by a human researcher orchestrating three AI systems:

| Phase | Versions | What Changed |
|-------|----------|--------------|
| **Compute** | v1–v2 | GPU brute-force verification, exhaustive attractor detection |
| **Explore** | v4–v6 | Operator algebra, invariant discovery, symbolic prediction |
| **Understand** | v7–v9 | Knowledge base (83 facts), causal chains, self-questioning |
| **Verify** | v10–v15 | Formal proofs (12/12), multi-base generalization, open questions |
| **Formalize** | M0–M4 | Canonical hashing, deterministic reproducibility, paper appendices |

The progression: *observing → classifying → predicting → proving*.

Full story: [Evolution from Scripts to Reasoning](EN/docs/EVOLUTION_FROM_SCRIPTS_TO_REASONING.md)

---

## 🎨 Visualizations & Essays

### [Convergence Pattern](assets/convergence-pattern.png)
High-resolution visualization of fixed-point clustering in digit-operation space.

### [The Mechanics of Emergence](assets/emergence-mechanisms.md)
Essay examining how simple rules create complex structure across five systems — from molecules to culture. Shows digit-dynamics as an instance of universal emergence patterns.

---

## 🔬 Key Results

### Mathematical Results (Paper A)

| Theorem | Statement | Scope |
|---------|-----------|-------|
| Symmetric FP count | (b−2) · b^(k−1) symmetric fixed points among 2k-digit numbers | All bases b ≥ 3 |
| Universal 1089-family | A_b = (b−1)(b+1)² generalizes 1089 to every base | All bases b ≥ 3 |
| Four infinite families | Explicit counting formulas, pairwise disjoint | Base 10 |
| Fifth family (1089-trick) | n_k = 110 · (10^(k−3) − 1) for k ≥ 5 | Base 10 |
| Kaprekar constants | K_b = (b/2)(b²−1) for even bases; 495 and 6174 algebraically | Bases b ≥ 4 |
| Armstrong upper bound | k_max(b) ≤ ⌊b · log(b) / log(b − 1)⌋ + 1 | All bases b ≥ 3 |
| Conditional Lyapunov | Digit-sum descent for operations in class P ∪ C | All bases |

### Computational Verification

- 260 unit tests across M0–M4 modules (deterministic infrastructure)
- 98 legacy tests across research engines (v4–v15)
- 12/12 algebraic proofs computationally verified
- Exhaustive verification over all k-digit inputs for k = 3…7
- Canonical SHA-256 hash chain: registry → pipeline → domain → result

---

## 🤖 Multi-Agent Research Process

This project used a tripartite collaboration model:

| Role | Agent | Contribution |
|------|-------|-------------|
| **Human Visionary** | R. Havenaar | Research direction, conceptual leaps, orchestration, algebraic insight |
| **Mathematical Consultant** | DeepSeek (R1–R5) | Deep mathematical reasoning, conjecture refinement |
| **Implementation & Scaling** | Manus (R6) | Bulk implementation, multi-base engine, protocol execution |
| **Formal Proofs & Architecture** | Claude/Cascade (R7–R11) | Proof verification, M0–M4 architecture, publication preparation |

The human researcher directed every research phase, identified the algebraic structures, and made the conceptual leaps connecting digit operations to modular arithmetic. The AI systems executed, verified, and formalized.

---

## 🏗️ Architecture

The codebase has two tracks:

### Research Engine (v15)
Single-file exploration engine (~6,500 lines). Contains 30 modules spanning 6 reasoning layers — from empirical dynamics to abductive reasoning. Used for discovery and conjecture generation.

### Reproducibility Infrastructure (M0–M4.1)
Modular, deterministic, submission-quality codebase:

| Module | Function | Lines |
|--------|----------|-------|
| **M0** (pipeline_dsl.py) | Canonical semantics, operation registry, SHA-256 identity | ~1,050 |
| **M1** (experiment_runner.py) | SQLite result store, batch execution, JSON export | ~640 |
| **M2** (feature_extractor.py) | Number features, orbit analysis, conjecture mining | ~900 |
| **M3** (proof_engine.py) | Proof skeletons, density estimation, ranking model v1.0 | ~1,160 |
| **M4** (appendix_emitter.py) | LaTeX appendix generation, manifest, reproducibility bundle | ~1,170 |

Key design decision: **Layer A (semantic) / Layer B (execution) separation** in M0. Pipeline specifications are pure data — inspectable, hashable, and independent of implementation.


---

## 🧪 Methodological Note

This research employed Production-Validation (P↔V) modularization as an organizing heuristic. The v1→v15 evolution exhibits 6 measurable P→V cycles (see [PV Methodology](EN/docs/PV_METHODOLOGY.md)).

**Important Disclaimers:**

This does NOT constitute proof that P↔V is:
- Mathematically necessary for digit-operation research
- Universal across all discovery domains  
- Superior to all alternative methodologies

What it DOES demonstrate:
- ✅ P↔V was **instrumentally useful** in this case
- ✅ Systematic knowledge accumulation (83 facts)
- ✅ Reusable proof modules (M0-M4)
- ✅ Measurable meta-oscillation (6 cycles)

**Efficiency Note:** Estimated ~4x speedup vs. random brute-force is NOT empirically validated. See [Limitations](EN/docs/LIMITATIONS.md) for full epistemic boundaries.

**Positioning:** This is a strong case study of P↔V utility in one domain, not proof of universality.

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@misc{syntriad2026digit,
  title={Algebraic Structure of Fixed Points in Composed Digit-Operation Dynamical Systems},
  author={Havenaar, Remco and SYNTRIAD Research},
  year={2026},
  note={Computational exploration of digit-operation pipelines across arbitrary bases},
  url={https://github.com/SYNTRIAD/digit-dynamics}
}
```

---

## 📜 License

MIT License — see [LICENSE](EN/LICENSE) for details.

---

## 🔗 Links

- **Papers:** [EN/papers/](EN/papers/)
- **Documentation:** [EN/docs/](EN/docs/)
- **Source Code:** [EN/src/](EN/src/) (M0–M4 modules)
- **Research Engines:** [EN/engines/](EN/engines/) (v1–v15)
- **Reproducibility:** [EN/src/reproduce.py](EN/src/reproduce.py)
- **SYNTRIAD Research:** [syntriad.com](https://syntriad.com)

---

*SYNTRIAD Research — February 2026*

