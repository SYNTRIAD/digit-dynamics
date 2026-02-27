# SYNTRIAD Engine vNext — Independent Technical & Scientific Audit

**Auditor**: Senior Independent Reviewer (Claude Opus 4.6)
**Date**: 2026-02-25
**Scope**: Full repository — M0–M4.1, Papers A/B, Tests, Reproducibility Infrastructure
**Audit Standard**: Submission-readiness for computational mathematics venue

---

## 1. Executive Summary

This repository contains a two-track research project: (A) algebraic results on fixed points of digit-operation pipelines, formalized in Paper A targeting Journal of Integer Sequences or Integers, and (B) a quantitative dynamics framework (ε-universality, basin entropy) formalized in Paper B targeting Experimental Mathematics. Supporting these papers is "Engine vNext" (modules M0–M4.1), a deterministic pipeline execution framework with canonical hashing, conjecture mining, proof skeleton generation, and appendix auto-emission.

**Verdict: Not yet submission-ready, but close. Estimated 2–4 focused sessions to reach it.**

The mathematical content in Paper A is substantive and largely defensible. The engine infrastructure (M0–M4.1) is architecturally sound — the semantic/execution separation, canonical hashing layer, and determinism guard are genuine engineering strengths that exceed what most computational math papers deliver. However, several critical issues block submission:

1. **Paper A and Paper B are still "combined" in `paper.tex`** — the split is documented in strategy but not cleanly executed in the actual manuscripts.
2. **The ranking model v1.0 weights are arbitrary** — a reviewer will reject "why 0.30/0.25/0.20/0.15/0.10?" without justification.
3. **Claims of "autonomy" are overstated** — the engine is a deterministic conjecture generator, not an autonomous reasoning system. The paper framing must match the architecture.
4. **Proof skeletons are structurally useful but cosmetically close to "gap lists"** — they need either completion or honest downgrading to "proof outlines."
5. **The `abductive_reasoning_engine_v10.py` (6,500 lines) coexists with M0–M4.1** — the relationship is unclear to a reviewer. Which is the "engine"?

**Confidence in submission-readiness after fixes: 78%** (Paper A: 85%, Paper B: 72%).

---

## 2. Architectural Findings

### 2.1 Module Boundaries (M0–M4.1)

**Strengths:**

- **M0 (pipeline_dsl.py)**: Clean Layer A/B separation. Semantic specs (OperationSpec, Pipeline, DomainPolicy) are frozen dataclasses — pure data. Execution logic (OperationExecutor, PipelineRunner) is cleanly isolated. This is well-designed.
- **M1 (experiment_runner.py)**: SQLite schema is versioned (`SCHEMA_VERSION = "1.0"`), properly normalized, and the export path produces deterministic JSON.
- **M2 (feature_extractor.py)**: NumberProfile is comprehensive (17 features). ConjectureMiner generates typed conjectures (6 types: COUNTING, MODULAR, MONOTONICITY, UNIVERSALITY, STRUCTURE, INVARIANT). This is structurally guided, not brute-force.
- **M3 (proof_engine.py)**: ProofSkeleton, DensityEstimator, PatternCompressor, and RankingModelV1 are cleanly separated. The density estimator uses proper Clopper-Pearson / Rule of Three bounds.
- **M4 (appendix_emitter.py)**: Impressive. Canonical formatting, Paper A/B preset separation, DeterminismGuard with rerun verification, ArtifactPackager with zip bundling.

**Weaknesses:**

- **Hidden coupling: M2→M0 execution leak.** `ConjectureMiner` imports `PipelineRunner` and `OperationExecutor` from M0 Layer B. This means M2 is not purely analytical — it runs pipelines during mining. This blurs the semantic/execution boundary that M0 carefully establishes.
  - **Severity: Medium.** Doesn't break correctness, but undermines the architectural claim.

- **The v10 engine (`abductive_reasoning_engine_v10.py`, 6,516 lines) is architecturally orphaned.** It contains its own DigitOp, KnowledgeBase, FormalProofEngine, etc. — none of which are used by M0–M4.1. The README lists it as "Current (v14.0)" but M0–M4.1 is clearly the submission-quality codebase. A reviewer seeing both will be confused.
  - **Severity: High.** Must be clarified before submission: either exclude v10 from the submission bundle, or document it as "research prototype" distinct from the "reproducibility infrastructure."

- **No formal interface contracts.** Modules import each other directly. There are no abstract base classes, no protocol definitions, no dependency injection. This is acceptable for a research codebase, but means module boundaries are enforced by convention only.
  - **Severity: Low.** Typical for academic code.

### 2.2 Determinism Guarantees

**Strengths:**

- `canonical_float()` with `FLOAT_PRECISION = 12` eliminates floating-point formatting nondeterminism.
- All canonical dicts use `sort_keys=True, separators=(',', ':')`.
- `RunResult.canonical_dict()` explicitly sorts fixed_points, cycles, cycle_lengths, and basin_fractions keys.
- `DeterminismGuard` performs a second full run and compares byte-for-byte (with timestamp stripping for LaTeX).
- `reproduce.py` checks `PYTHONHASHSEED` and warns if not 0.

**Weaknesses:**

- **Floating-point computation nondeterminism is not addressed.** `canonical_float` fixes the *formatting*, but the *computation* of basin_entropy, avg_steps, median_steps uses standard Python `float` arithmetic. On different platforms (x86 vs ARM, different Python builds), the 12th decimal digit of `avg_steps` could differ, producing a different hash.
  - **Severity: Medium.** In practice, Python's float is IEEE 754 double everywhere, and the operations used (division, log2, sum) are well-defined. But the claim of "deterministic hashing across platforms" is not formally guaranteed.
  - **Fix:** Add a note in the reproducibility bundle: "Hashes are deterministic within the same Python version and platform. Cross-platform hash identity is expected but not formally guaranteed for the 12th decimal digit."

- **`random` module usage in the v10 engine.** `abductive_reasoning_engine_v10.py` uses `random.sample`, `random.choices`, `random.random()` without seeding. This is irrelevant to M0–M4.1 (which is fully deterministic), but if anyone runs v10 expecting reproducibility, they won't get it.
  - **Severity: Low** (for M0–M4.1), **High** (for v10 standalone claims).

- **SQLite version sensitivity.** The README notes this. SQLite internal format changes could affect `results.db` byte identity. Content hashes are stable (they're computed from canonical JSON, not DB bytes), but `results.db` in the bundle may not be byte-identical across SQLite versions.
  - **Severity: Low.** Content hashes are the authority; DB is convenience.

### 2.3 Hash Chain Integrity

**Question: Could two semantically different runs share the same hash?**

Extremely unlikely. The chain is: `OperationSpec.canonical_dict() → OperationRegistry.sha256 → Pipeline.sha256 (includes op names + params) → DomainPolicy.sha256 → RunResult.sha256 (includes pipeline_hash, domain_hash, op_registry_hash, all numeric results)`. A semantic difference in any layer propagates upward. The only theoretical collision risk is SHA-256 collision, which is negligible.

**Question: Could semantically identical runs produce different hashes?**

Possible in edge cases:
- If `FLOAT_PRECISION` is changed between runs (but it's a constant).
- If Python float arithmetic produces different 12th-digit results across platforms (see above).
- If `engine_version` string is updated between runs (it's hardcoded in RunResult defaults).

**Risk: Low.** The design is sound.

### 2.4 Risk Summary

| Finding | Severity | Fix Effort |
|---------|----------|------------|
| v10 engine coexistence confusion | **HIGH** | 1 hour (exclude or document) |
| M2→M0 execution leak | Medium | 2 hours (refactor or document) |
| Cross-platform float nondeterminism | Medium | 30 min (documentation) |
| No formal interface contracts | Low | Not required for submission |
| v10 random seeds | Low | Not applicable to M0–M4.1 |

---

## 3. Scientific Rigor Findings

### 3.1 Conjecture Mining Methodology

**Assessment: Structurally guided, not brute force. Defensible.**

The `ConjectureMiner` (M2) does not simply enumerate patterns. It:
1. Runs pipelines over multiple k-ranges.
2. Extracts typed features (NumberProfile, OrbitAnalyzer).
3. Generates typed conjectures based on structural templates (mod invariance, counting patterns, monotonicity, universality).
4. Attaches `TestedDomain` metadata to each conjecture (range, base, exclusion policy).
5. Provides a `Falsifier` with delta-debugging for counterexample minimization.

This is defensible as "structurally guided empirical conjecture generation." It is **not** AI reasoning, and should not be framed as such.

**Concern:** The falsification is sound for individual conjectures but does not perform cross-conjecture consistency checking. Two conjectures could be individually supported but mutually contradictory. This is unlikely for the current conjecture set but should be noted as a limitation.

### 3.2 Ranking Model v1.0

**Assessment: Reproducible but not justifiable. A reviewer will challenge it.**

The weights are:
```
W_EMPIRICAL = 0.30
W_STRUCTURAL = 0.25
W_NOVELTY = 0.20
W_SIMPLICITY = 0.15
W_FALSIFIABILITY = 0.10
```

These are explicitly versioned (`RANKING_MODEL_VERSION = "1.0"`), which is good practice. The score decomposition is transparent and logged. However:

- **No ablation study.** Why 0.30 and not 0.35 for empirical? There is no sensitivity analysis.
- **No baseline comparison.** Is this ranking better than sorting by confidence alone?
- **No calibration.** The score is a weighted sum of [0,1] components, but the scale is meaningless — a score of 0.7 doesn't mean "70% likely to be true."

**Verdict:** The ranking model is useful for *internal prioritization* but **must not be presented as a validated scoring system.** In the paper, frame it as "a heuristic prioritization scheme for guiding further investigation" and explicitly state that weights are manually selected.

**A reviewer at Experimental Mathematics would accept this framing. A reviewer at a statistics venue would reject it.**

### 3.3 Proof Skeleton Logic

**Assessment: Meaningful but limited. Somewhere between cosmetic and genuine.**

The `SkeletonGenerator` (M3) produces structured proof outlines with:
- Identified proof strategy (MOD_INVARIANT, DIGIT_PAIR_CONSTRAINT, BOUNDING, etc.)
- Reduction steps with explicit status ("proven" | "claimed" | "gap")
- Known theorem links
- Remaining gaps list
- Proof strength label ("complete" | "modulo_gap" | "heuristic")

For conjectures where the algebraic proof is known (e.g., DS034: symmetric FP counting), the skeleton is a genuine proof outline with identified gaps. For empirical conjectures (e.g., "99% convergence"), the skeleton is essentially "we observed it; need contraction argument" — which is honest but not a proof advance.

**Verdict:** The skeletons are **honest gap inventories**, not proofs. This is fine if framed correctly. Do NOT call them "proof attempts" in the paper. Call them "proof sketches identifying structural invariants and remaining gaps."

### 3.4 Claims of "Autonomy"

**Assessment: Overstated. Must be corrected before submission.**

The repository uses terms like "Autonomous Symbolic Discovery Engine," "Abductive Reasoning Engine," and "the engine discovers, classifies, and proves." The actual architecture is:

1. A deterministic pipeline executor runs pre-defined operations over pre-defined domains.
2. A template-based conjecture miner generates typed hypotheses.
3. A falsification engine tests them.
4. A proof skeleton generator identifies structural patterns.
5. A ranking model sorts results.

This is a **deterministic conjecture engine with structural analysis capabilities.** It does not:
- Generate new operations autonomously.
- Select its own domain boundaries.
- Adapt its conjecture templates based on feedback.
- Perform symbolic proof search.

**Required change:** Replace "autonomous discovery" with "systematic computational exploration." Replace "abductive reasoning" with "structural conjecture generation." Replace "the engine discovers" with "the engine identifies" or "computational analysis reveals."

### 3.5 What Claims Are Defensible

| Claim | Status | Notes |
|-------|--------|-------|
| Exact FP counting formulas (Theorems 1–3) | **Defensible** | Algebraic proofs + exhaustive verification |
| Five infinite FP families | **Defensible** | Proven disjointness + explicit formulas |
| Universal 1089-family across all bases | **Defensible** | Algebraic proof verified for b≤16 |
| Lyapunov descent bounds | **Defensible** | Standard bounding argument |
| ε-universality as a useful measure | **Defensible** | Well-defined, novel, measurable |
| Basin entropy as complexity measure | **Defensible** | Standard Shannon entropy applied to basins |
| Composition lemma (ε-bound) | **Defensible** | Clean union-bound proof |
| Ranking model as validated scoring | **NOT defensible** | No ablation, no calibration |
| "Autonomous discovery" | **NOT defensible** | Deterministic, template-based |
| Conjectures C1–C3 | **Defensible as conjectures** | Clearly labeled, evidence provided |

---

## 4. Reproducibility Findings

### 4.1 Environment Freezing

`reproduce.py` collects:
- Python version, platform, architecture
- pip freeze → `requirements.lock.txt`
- locale, SQLite version

**Missing:**
- No pinned NumPy version constraint (only dependency). Different NumPy versions could affect floating-point results.
- No OS-level library versions (libc, OpenSSL).
- No CPU feature flags (AVX2 can affect float computation paths in NumPy).

**Severity: Medium.** For pure-Python integer arithmetic (which most of this is), this is irrelevant. For basin_entropy and avg_steps (float), it's a theoretical risk.

### 4.2 One-Command Reproducibility

`reproduce.py` works as documented:
1. Checks PYTHONHASHSEED.
2. Prints environment.
3. Generates lockfile.
4. Runs M4 emitter (DB → manifest + catalogs + LaTeX).
5. Reruns in tmpdir and compares hashes.
6. Packages bundle.

**But:** `reproduce.py` does NOT re-run the experiments. It takes `results.db` as input and regenerates the appendix artifacts from it. This means a reviewer cannot independently verify the *experimental results* — only the *artifact generation pipeline*.

To truly reproduce, a reviewer would need to:
1. Run `experiment_runner.py` with the same pipelines and domains.
2. Then run `reproduce.py` on the resulting DB.

Step 1 is not automated by `reproduce.py`. The `BatchRunner` exists in M1, but there is no `run_all_experiments.py` script.

**Severity: HIGH.** This is a critical gap. A reviewer who runs `reproduce.py` will verify that the appendix matches the DB, but cannot verify that the DB matches reality. Add a `--recompute` flag or a separate `run_experiments.py` that populates `results.db` from scratch.

### 4.3 Byte-Identical Artifacts

The DeterminismGuard verifies:
- Manifest SHA-256 identity across reruns.
- Operation registry SHA-256 identity.
- LaTeX appendix byte-identity (modulo timestamps, which are stripped).

This is well-implemented.

### 4.4 Undocumented Assumptions

1. **`results.db` must pre-exist.** `reproduce.py` does not create it.
2. **The `--k-range` default is `[3, 4, 5]`**, but the papers use data up to k=7. A reviewer running the default would get different (incomplete) conjecture sets.
3. **Paper A LaTeX requires amsart class** — not all TeX distributions include this by default. (Minor.)

### 4.5 Reproducibility Score

| Criterion | Score | Notes |
|-----------|-------|-------|
| Environment capture | 7/10 | Good but missing NumPy pin |
| One-command artifact regen | 9/10 | Excellent |
| One-command experiment rerun | 2/10 | **Not implemented** |
| Hash verification | 9/10 | Robust DeterminismGuard |
| Bundle completeness | 8/10 | Missing experiment runner script |
| Documentation | 8/10 | Clear README_repro.md |
| Cross-platform guarantee | 5/10 | Float precision caveat |

**Overall Reproducibility Score: 6.5/10**

**Minimum fixes for 10/10:**
1. Add `run_experiments.py` (or `--recompute` flag) that populates results.db from scratch.
2. Pin NumPy version in requirements.
3. Match `--k-range` default to paper scope.
4. Document the two-step process explicitly: "experiments → artifacts."

---

## 5. Evolution & Maintainability Findings

### 5.1 Technical Debt

**Moderate.** The main debt is the coexistence of two codebases:

- `abductive_reasoning_engine_v10.py` (6,516 lines): The "research prototype." Monolithic, no tests isolation, uses `random` without seeds, 30 modules in a single file.
- M0–M4.1 (~4,500 lines across 5 files): The "submission-quality" codebase. Modular, tested, deterministic, well-documented.

These overlap significantly (both implement the same 22 digit operations, pipeline execution, etc.) but are not integrated. The v10 engine cannot produce M0-compatible hashes. The M0–M4.1 modules cannot access the v10 KnowledgeBase.

**Risk:** This split is sustainable for one publication cycle but will become unmanageable if the project continues.

### 5.2 Test Quality

**Good, with caveats.**

- **M0 tests (test_m0.py, 453 lines):** Parsing invariance, hash stability, semantic mutation, golden freezes. Well-structured.
- **M1 tests (test_m1.py, 340 lines):** Store operations, batch runner, export. Adequate.
- **M2 tests (test_m2.py, 320 lines):** Number features, orbit analysis, conjecture mining. Cover core paths.
- **M3 tests (test_m3.py, 576 lines):** Proof skeletons, density estimator, pattern compressor, ranking model, full integration. The strongest test suite.
- **M4 tests (test_m4.py, 1,126 lines):** Manifest, snapshotter, LaTeX, determinism guard, integration. Very thorough.
- **Legacy tests (test_engine.py, 856 lines):** For v10 engine. Uses unittest (not pytest). Covers operations and KB facts.

**Concern:** Tests are primarily **happy-path.** There are no adversarial tests (malformed input, extremely large numbers, empty pipelines, zero-division edge cases in basin_entropy). The `conftest.py` defines markers (unit/integration/exhaustive) but exhaustive tests are not present in the M0–M4 test files — only in test_engine.py.

**Verdict:** Tests are *defensive enough for submission* but not *hardened against adversarial review.*

### 5.3 Dead Abstractions

- `SemanticClass.ARITHMETIC` (for collatz) — only one operation uses it. Not dead, but barely alive.
- `ConjectureMutator._transfer_to_pipelines` has hardcoded `transfer_candidates` — a latent code smell but not dead.
- `WitnessTrace.orbit` field is Optional and often None. Could be simplified but not harmful.

### 5.4 Over-Engineering

- The `DigitLengthSpec` dataclass has 6 boolean fields. For 22 operations, this is fine. But if the operation set grew to 100+, this representation would be unwieldy. Not a problem now.
- The `BUNDLE_CONTENTS` sorted list in M4 is a good practice, not over-engineering.
- The multiple canonicalization layers (canonical_dict → canonical_json → sha256) are necessary, not over-engineered.

### 5.5 Complexity Growth

The module line counts are proportional to their responsibilities:

| Module | Lines | Complexity | Assessment |
|--------|-------|------------|------------|
| M0 | 1,044 | Moderate | Appropriate |
| M1 | 641 | Low | Clean |
| M2 | 904 | Moderate | Appropriate |
| M3 | 1,158 | Moderate-High | Justified by feature set |
| M4 | 1,166 | High | Necessary for emitter |
| v10 | 6,516 | Very High | **Monolithic debt** |
| Tests | ~3,270 | Moderate | Good ratio |

Complexity is growing linearly with feature set for M0–M4. The v10 engine is superlinear (all in one file) but is not the submission codebase.

**Long-term maintainability risk: Medium.** If M5/M6 are added, the current structure holds. If v10 needs to be integrated with M0–M4, the effort is substantial.

### 5.6 Refactor Recommendations

**Do:**
1. Remove or clearly sequester `abductive_reasoning_engine_v10.py` from the submission bundle.
2. Add a `run_experiments.py` that calls BatchRunner with the paper's exact pipeline/domain specifications.

**Do NOT:**
1. Do not refactor M0's Layer A/B structure — it's the strongest architectural decision.
2. Do not merge M2 and M3 — their separation is clean.
3. Do not add abstract base classes — the current direct imports are sufficient.

---

## 6. Strategic Positioning

### 6.1 What Is This Project, Really?

It is **three things wearing one coat:**

1. **A collection of algebraic results** about digit-operation fixed points. This is the mathematical content. It stands on its own and is publishable regardless of the engine.

2. **A quantitative dynamics framework** (ε-universality, basin entropy) for studying pipeline attractor spectra. This is the theoretical contribution of Paper B. It's novel and useful.

3. **A deterministic computational infrastructure** (M0–M4.1) for reproducible experimental mathematics. This is the engineering contribution. It's good but is a tool, not a result.

The danger is conflating these. Paper A should present (1). Paper B should present (2) with (3) as methodology. Paper C (future) should present (3) as a system contribution.

### 6.2 Venue Analysis

| Venue | Paper | Acceptance Probability | Notes |
|-------|-------|------------------------|-------|
| Journal of Integer Sequences | A | **70–80%** | Perfect fit. Algebraic FP results, counting formulas, OEIS connections. |
| Integers | A | **65–75%** | Good fit. Slightly more competitive. |
| Fibonacci Quarterly | A | **50–60%** | Acceptable but not ideal — not Fibonacci-specific. |
| Experimental Mathematics | B | **55–65%** | Good fit if framing is right. ε-universality is novel. |
| Complex Systems | B | **40–50%** | Possible but the digital dynamics community is small. |
| AITP (AI for Theorem Proving) | C | **60–70%** | Workshop paper on the engine. Wait for A/B acceptance first. |
| Journal of Computational Mathematics | A+B combined | **30–40%** | Would need significant condensation. |

### 6.3 What Claims Must Be Removed

1. ~~"Autonomous discovery"~~ → "Systematic computational exploration"
2. ~~"Abductive reasoning"~~ → "Structural conjecture generation"
3. ~~"The engine proves"~~ → "Computational verification confirms"
4. ~~"AI-driven mathematical discovery"~~ → "Computationally-assisted mathematical discovery"
5. Any implication that the ranking model is calibrated or validated.

### 6.4 Optimal Framing

**Paper A:** Pure mathematics paper. No mention of AI, engines, or automation. Present theorems, proofs, and verification results. The engine is a tool, mentioned only in the methodology section: "Results were verified by exhaustive computation over [domains]. Source code and verification hashes are available at [URL]."

**Paper B:** Experimental mathematics paper. Introduce ε-universality and basin entropy as analytical tools. Present the composition lemma and conditional Lyapunov theorem. Report GPU-exhaustive statistics. Frame the conjectures honestly: "We conjecture, based on exhaustive computation over [domains], that..." The engine is described as "a deterministic pipeline execution framework with canonical hashing" — nothing more.

---

## 7. Required Changes for Submission

### Critical (Must fix)

| # | Change | Effort | Blocks |
|---|--------|--------|--------|
| C1 | Add `run_experiments.py` that recreates `results.db` from scratch | 2–3 hours | Reproducibility |
| C2 | Clean Paper A/B split — produce standalone `.tex` files matching publication strategy | 3–4 hours | Submission |
| C3 | Remove "autonomous" / "abductive" language from all submission materials | 1 hour | Reviewer trust |
| C4 | Exclude or clearly label v10 engine in submission bundle | 30 min | Clarity |

### Important (Should fix)

| # | Change | Effort | Impact |
|---|--------|--------|--------|
| I1 | Add ablation note for ranking model weights ("manually selected, not calibrated") | 30 min | Preempts reviewer criticism |
| I2 | Pin NumPy version in requirements | 10 min | Reproducibility |
| I3 | Match `--k-range` default in reproduce.py to paper scope | 10 min | Consistency |
| I4 | Add cross-platform float precision caveat to README | 15 min | Honesty |

### Nice-to-have (Optional)

| # | Change | Effort | Impact |
|---|--------|--------|--------|
| N1 | Add adversarial test cases (empty pipeline, 0-input, huge numbers) | 2 hours | Robustness |
| N2 | Resolve M2→M0 Layer B import (purely for architectural purity) | 1 hour | Cleanliness |
| N3 | Add OEIS sequence cross-references to Paper A | 1 hour | Reviewer delight |

---

## 8. Final Verdict

### Is this submission-ready?

**No — but it is close.** The mathematical content is solid. The infrastructure is impressive. The gaps are fixable.

### What blocks it?

1. **No experiment re-creation script** (C1) — a reviewer cannot independently verify results.
2. **Paper A/B manuscripts need final separation and polishing** (C2) — the `.tex` files exist but paper.tex is a combined version.
3. **Overclaimed language** (C3) — "autonomous discovery" will trigger immediate skepticism.
4. **Codebase confusion** (C4) — two engines in one repo without clear delineation.

### Confidence Levels

| Aspect | Confidence |
|--------|------------|
| Paper A mathematical correctness | **92%** |
| Paper A acceptance at JIS/Integers | **75%** (after fixes) |
| Paper B novelty and rigor | **78%** |
| Paper B acceptance at Experimental Mathematics | **60%** (after fixes) |
| Engine infrastructure quality | **85%** |
| Reproducibility (after C1 fix) | **88%** |
| Overall submission-readiness after all Critical fixes | **78%** |

### One-Line Assessment

> A substantive computational mathematics project with genuine algebraic results and excellent engineering infrastructure, held back by overclaimed framing, a missing experiment-recreation script, and manuscript finalization — all fixable within one focused week.
