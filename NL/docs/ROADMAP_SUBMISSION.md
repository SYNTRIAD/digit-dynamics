# Roadmap: Submission Pack — Audit Response & Action Plan

**Date**: 2026-02-25
**Source**: Independent Technical Audit (docs/SYNTRIAD_ENGINE_vNext_AUDIT_REPORT.md) + GPT meta-analysis
**Goal**: Address all critical blockers → submission-ready for Paper A (JIS/Integers) and Paper B (Experimental Mathematics)
**Estimated effort**: 5 focused sessions

---

## Current State Assessment

| Metric | Score | Source |
|--------|-------|--------|
| Paper A mathematical correctness | 92% | Audit §3.5 |
| Paper B novelty and rigor | 78% | Audit §8 |
| Engine infrastructure quality | 85% | Audit §8 |
| Reproducibility | 6.5/10 | Audit §4.5 |
| Overall submission-readiness | 78% (after C-fixes) | Audit §8 |

### What the audit explicitly validates

- Hash chain integrity → **sound**
- Determinism design → **strong**
- Proof skeletons → **honestly classified**
- Algebraic results (Theorems 1–3, five FP families) → **defensible**
- M0–M4 engine architecture → **solid**
- Composition lemma → **correct**
- ε-universality → **legitimate concept**

### What blocks submission

| ID | Finding | Severity | GPT Assessment |
|----|---------|----------|----------------|
| C1 | No experiment re-run script | **CRITICAL** | "artifact reproducibility but not result reproducibility" |
| C2 | Paper A/B split not cleanly executed | **CRITICAL** | "inconsistentie waar reviewers allergisch voor zijn" |
| C3 | Autonomy claims overstated | **CRITICAL** | "Reviewer #2 gaat je daarop fileren" |
| C4 | Two engines in one repo, no delineation | **HIGH** | "evolutionair chaos i.p.v. gecontroleerde architectuur" |

---

## Phase 0: Already Done (repo restructure)

✅ **C4 partially addressed** — repo restructured into:

```
autonomous_discovery_engine/
├── pipeline_dsl.py          # M0 core
├── experiment_runner.py     # M1 core
├── feature_extractor.py     # M2 core
├── proof_engine.py          # M3 core
├── appendix_emitter.py      # M4 core
├── conftest.py              # pytest config
├── tests/                   # all test files (357+ tests)
├── engines/                 # v9, v10, and other standalone engines
├── scripts/                 # reproduce.py, gpu scripts, analysis
├── papers/                  # paper.tex, paper_A.tex, paper_B.tex, PDFs
├── docs/                    # documentation, audit report, trajectory
└── data/                    # .db files, JSON exports
```

The v10 engine (6,516 lines) is now physically separated in `engines/`.
M0–M4 core modules stay at root as the submission codebase.

**Remaining for C4**: Exclude `engines/` from the reproducibility bundle. Add a one-line note to `README.md` explaining the split.

---

## Phase 1: C1 — Experiment Re-Creation Script (Day 1)

### Problem
`reproduce.py` regenerates artifacts from `results.db`, but does not recreate the experiments themselves. A reviewer cannot verify that the DB matches reality.

### Solution
Create `scripts/run_experiments.py` that populates `results.db` from scratch using the exact pipeline/domain specs used in the papers.

### Building blocks already in place
- `experiment_runner.py` already has `kaprekar_suite()`, `paper_b_suite()`, `truc1089_suite()` returning the exact (pipeline, domain) pairs
- `experiment_runner.py __main__` already runs all three suites sequentially
- `BatchRunner` handles all the actual execution + SQLite storage

### Tasks

1. **Create `scripts/run_experiments.py`**
   - Accept `--db` (output path, default `data/results.db`)
   - Accept `--suites` (choice of `kaprekar`, `truc1089`, `paper_b`, `all`; default `all`)
   - Accept `--k-range` to allow k=3..7 (matching paper scope)
   - Delete existing DB if `--fresh` flag
   - Run all suites via BatchRunner
   - Print summary + export `results_export.json` and `paper_b_hashes.json`
   - Exit 0 on success

2. **Add Paper A-specific suite**
   - `paper_a_suite()` in experiment_runner.py: Kaprekar k=3..7 + truc_1089 k=3..7 using `DomainPolicy.paper_a_kaprekar` and `DomainPolicy.paper_a_1089` presets
   - This uses the exact domains Paper A claims results for

3. **Update `scripts/reproduce.py`**
   - Add `--recompute` flag that calls `run_experiments.py` first
   - Document the two-step process: experiments → artifacts

4. **Update `README_repro.md` template** (in appendix_emitter.py)
   - Document: `python scripts/run_experiments.py --db data/results.db --fresh`
   - Then: `python scripts/reproduce.py --db data/results.db --out repro_out --bundle`

5. **Fix `--k-range` default**
   - Change default from `[3, 4, 5]` to `[3, 4, 5, 6, 7]` to match paper scope (audit I3)

### Acceptance criteria
- `python scripts/run_experiments.py --fresh` produces a `results.db` from scratch
- `python scripts/reproduce.py --db data/results.db --bundle` then produces identical artifacts
- Round-trip: fresh DB → emit → guard → all green

---

## Phase 2: C3 — Language Correction (Day 2)

### Problem
Terms like "Autonomous Discovery Engine", "Abductive Reasoning", "the engine discovers" are overclaimed. The architecture is a deterministic conjecture generator, not an autonomous reasoning system.

### Required replacements (audit §3.4, §6.3)

| Old | New |
|-----|-----|
| "Autonomous discovery" | "Systematic computational exploration" |
| "Abductive reasoning" | "Structural conjecture generation" |
| "The engine discovers" | "Computational analysis identifies" |
| "The engine proves" | "Computational verification confirms" |
| "AI-driven mathematical discovery" | "Computationally-assisted mathematical discovery" |

### Files to update

1. **`papers/paper_A.tex`** — methodology section only; Paper A should mention the engine minimally: "Results were verified by exhaustive computation. Source code and verification hashes available at [URL]."
2. **`papers/paper_B.tex`** — describe engine as "a deterministic pipeline execution framework with canonical hashing" in methodology section
3. **`README.md`** — replace "Autonomous Discovery Engine" in title/description with "Computational Exploration Engine" or similar
4. **`docs/DISCOVERY_ENGINES_DOCUMENTATION.md`** — update terminology
5. **Do NOT rename files or modules** — internal code identifiers are not submission materials

### What is defensible (keep as-is)
- "Structurally guided empirical conjecture generation" ✅
- "Deterministic conjecture engine with structural analysis" ✅
- "Proof sketches identifying structural invariants and remaining gaps" ✅
- "Heuristic prioritization scheme for guiding further investigation" ✅ (ranking model)

---

## Phase 3: C2 — Paper A/B Manuscript Finalization (Day 2–3)

### Problem
`paper.tex` is a combined version. `paper_A.tex` and `paper_B.tex` exist but may not be fully standalone submission-ready manuscripts.

### Current state
- `paper.tex` (775 lines): "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases: Algebraic Structure, Four Infinite Families, and Universality" — combined
- `paper_A.tex` (727 lines): "Fixed Points of Digit-Operation Pipelines in Arbitrary Bases: Algebraic Structure and Five Infinite Families" — algebraic focus
- `paper_B.tex` (521 lines): "Attractor Spectra and ε-Universality in Digit-Operation Dynamical Systems" — dynamics focus

### Tasks

1. **Verify Paper A is self-contained**
   - Must compile standalone: `pdflatex paper_A.tex` (in papers/)
   - Must not reference Paper B results
   - Must not use engine/AI terminology (C3)
   - Must contain: Theorems 1–3, five FP families, counting formulas, algebraic proofs
   - Methodology section: "Verified by exhaustive computation over [domains]"

2. **Verify Paper B is self-contained**
   - Must compile standalone: `pdflatex paper_B.tex`
   - Must not depend on Paper A theorems (may reference as "[companion paper]")
   - Must contain: ε-universality definition, basin entropy, composition lemma, conditional Lyapunov theorem
   - Must frame engine as methodology, not contribution
   - Conjectures clearly labeled: "We conjecture, based on exhaustive computation over [domains], that..."

3. **Add ablation note for ranking model** (audit I1)
   - In Paper B methodology section: "Weights are manually selected for internal prioritization and are not calibrated. See [supplementary] for sensitivity analysis."

4. **Deprecate `paper.tex`**
   - Add comment at top: `% DEPRECATED: See paper_A.tex and paper_B.tex for submission manuscripts`
   - Or remove entirely if Paper A and Paper B fully cover its content

---

## Phase 4: C4 — Submission Bundle Cleanup (Day 3)

### Problem (remaining)
The submission bundle should not include the v10 monolith or other non-essential files.

### Tasks

1. **Update `BUNDLE_CONTENTS` in `appendix_emitter.py`**
   - Ensure bundle does NOT include anything from `engines/`
   - Include `scripts/run_experiments.py` (C1 deliverable)
   - Include `scripts/reproduce.py`

2. **Add `engines/README.md`**
   - One paragraph: "This directory contains research prototypes developed during the exploration phase. These are NOT part of the submission codebase. The submission engine consists of M0–M4 modules at the project root."

3. **Update root `README.md`**
   - Clear section: "Repository Structure" explaining what's where
   - Clear section: "Submission Codebase" pointing to M0–M4 only
   - Clear section: "Reproduction" pointing to the two-step process

---

## Phase 5: Important Fixes (Day 4)

### I1 — Ranking model ablation note ✅ (covered in Phase 3)

### I2 — Pin NumPy version
- Add `numpy>=1.24,<2.0` to requirements in reproduce.py lockfile generation
- Or add a `requirements.txt` at project root with pinned versions

### I3 — Match k-range default ✅ (covered in Phase 1)

### I4 — Cross-platform float precision caveat
- Add to `README_repro.md` template:
  > "Hashes are deterministic within the same Python version and platform. Cross-platform hash identity is expected but not formally guaranteed for floating-point values beyond the 12th decimal digit."

### I5 — Document M2→M0 execution leak (from audit §2.1)
- Add comment in `feature_extractor.py` at the import site:
  > "Note: ConjectureMiner imports PipelineRunner for empirical evaluation during mining. This is a pragmatic integration point, not a Layer A/B violation."

---

## Phase 6: Nice-to-Have (Day 5, if time permits)

### N1 — Adversarial test cases
- Empty pipeline, 0-input, single-digit domain, base=2 edge cases
- Malformed manifest in DeterminismGuard
- These are defensive but not blocking

### N2 — M2→M0 Layer B refactor
- Extract pipeline execution from ConjectureMiner into a separate evaluation callback
- Not required for submission

### N3 — OEIS cross-references in Paper A
- FP counting formulas → search OEIS for matching sequences
- Strengthens Paper A considerably for JIS audience
- "Reviewer delight" factor

---

## Acceptance Criteria (Definition of Done)

### Reproducibility (target: 10/10)
- [ ] `python scripts/run_experiments.py --fresh` creates results.db from scratch
- [ ] `python scripts/reproduce.py --db data/results.db --bundle` produces verified bundle
- [ ] DeterminismGuard: 0 issues
- [ ] Round-trip rerun: byte-identical artifacts
- [ ] k-range default matches paper scope (3..7)
- [ ] NumPy pinned in requirements
- [ ] Cross-platform caveat documented

### Paper A
- [ ] Compiles standalone (pdflatex paper_A.tex)
- [ ] No engine/AI/autonomy language
- [ ] Self-contained algebraic results
- [ ] Methodology section references "exhaustive computation"
- [ ] OEIS cross-references (N3, optional)

### Paper B
- [ ] Compiles standalone (pdflatex paper_B.tex)
- [ ] Engine described as "deterministic pipeline framework" only
- [ ] Ranking model framed as "heuristic prioritization"
- [ ] Conjectures clearly labeled as empirical
- [ ] Ablation note for ranking weights

### Repo & Bundle
- [ ] `engines/` excluded from submission bundle
- [ ] `engines/README.md` explains the split
- [ ] Root `README.md` has clear structure documentation
- [ ] All 357+ tests passing
- [ ] No "autonomous"/"abductive" language in submission materials

---

## Priority Order

```
Day 1:  C1 (run_experiments.py) + I3 (k-range fix)
Day 2:  C3 (language correction) + I1 (ablation note)
Day 3:  C2 (paper A/B finalization) + C4 (bundle cleanup)
Day 4:  I2 (NumPy pin) + I4 (float caveat) + I5 (M2 doc) + test pass
Day 5:  N1–N3 (adversarial tests, OEIS refs) + final verification
```

---

## Strategic Notes

### The audit's implicit message (per GPT meta-analysis)

> There are 3 projects in 1 repo:
> 1. Algebraic wiskunde (Paper A)
> 2. Dynamische analyse (Paper B)
> 3. Engine-infrastructuur (M0–M4)
>
> Stop met ze tegelijk te verkopen.

**Action**: Paper A sells the math. Paper B sells the dynamics framework. The engine is a tool mentioned in methodology — not a contribution (yet). Paper C (engine as system contribution) comes after A and B are accepted.

### What NOT to do now
- Do NOT write Paper C
- Do NOT extend the ranking model
- Do NOT add new engine layers (M5/M6)
- Do NOT build agent algebra
- Do NOT start multi-base generalization work

### Post-submission roadmap (after acceptance)
1. Paper C: Engine as system contribution → AITP workshop
2. v10 integration or deprecation decision
3. Multi-base generalization (Q5–Q7 from open questions)
4. Agent algebra formalization (only with empirical foundation)

---

## Audit Response Summary

| Audit Finding | Response | Status |
|---------------|----------|--------|
| C1: No experiment rerun | Create `run_experiments.py` | **Phase 1** |
| C2: Paper A/B not clean | Finalize standalone manuscripts | **Phase 3** |
| C3: Overclaimed language | Systematic terminology replacement | **Phase 2** |
| C4: v10 coexistence | Repo restructured + bundle exclusion | **Phase 0 ✅ + Phase 4** |
| I1: Ranking ablation | Add note in Paper B | **Phase 3** |
| I2: NumPy pin | Add to requirements | **Phase 5** |
| I3: k-range default | Fix to [3..7] | **Phase 1** |
| I4: Float caveat | Document in README_repro | **Phase 5** |
| N1: Adversarial tests | Add edge-case tests | **Phase 6** |
| N2: M2→M0 refactor | Document, don't refactor | **Phase 5** |
| N3: OEIS refs | Add to Paper A | **Phase 6** |

**Target: submission-ready in 5 focused sessions.**
