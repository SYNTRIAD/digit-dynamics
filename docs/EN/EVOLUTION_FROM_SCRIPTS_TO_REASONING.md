# From Scripts to Reasoning

*The evolution of a brute-force calculator into a layered reasoning system*

---

## Overview

This document describes how a series of Python scripts for computing digit operations evolved into an autonomous research system with six layers of reasoning. The broader Mahler project has been running for about five weeks. But the autonomous discovery engine — from the first GPU script to a system with six reasoning layers, 30 modules, formal proofs, and the P_k discovery — emerged in **three days** (February 23-26, 2026). It is the story of a machine that learned — not in the AI-marketing sense of the word, but in the scientific sense: from computing to explaining to proving.

The evolution proceeded in four phases:

1. **Phase I: Computing** — GPU brute-force, processing millions of numbers
2. **Phase II: Discovering** — Recognizing patterns, generating pipelines, formulating hypotheses
3. **Phase III: Understanding** — Algebraic structure, proof sketches, knowledge base
4. **Phase IV: Justifying** — Formal modules, reproducible protocol, publication-ready

---

## Project Structure

```
mahler-analysis/
├── syntriad_extreme_experiments/       ← Phase 0: Mahler numerics
├── syntriad_theoretical_experiments/   ← Phase 0: Theoretical exploration
└── symmetry_discovery/                 ← Phase I-IV
    ├── symmetry_discovery_engine.py    ← v1.0: The first engine
    ├── run_discovery.py                ← Launcher
    ├── quick_research.py               ← Quick iteration
    ├── scoring_engine_v2.py            ← Triviality filter
    ├── meta_discovery_engine.py        ← v2.0: Self-improving
    ├── autonomous_researcher.py        ← v3.0: Dynamic operations
    ├── gpu_symmetry_hunter.py          ← GPU CUDA kernels
    ├── gpu_deep_researcher.py          ← GPU + self-adaptation
    ├── gpu_creative_research.py        ← GPU creative operations
    ├── extended_research_session.py    ← v4.0: Long-running autonomous
    ├── extended_research_session_v5.py ← v5.0: Cycles + genetic
    ├── extended_research_session_v6.py ← v6.0: ML-accelerated
    └── autonomous_discovery_engine/    ← Phase III-IV
        ├── engines/                    ← Research prototypes
        │   ├── autonomous_discovery_engine_v4.py   (36 KB)
        │   ├── meta_symmetry_engine_v5.py          (48 KB)
        │   ├── invariant_discovery_engine_v6.py     (56 KB)
        │   ├── symbolic_dynamics_engine_v7.py       (54 KB)
        │   ├── deductive_theory_engine_v8.py        (73 KB)
        │   ├── abductive_reasoning_engine_v9.py    (105 KB)
        │   └── abductive_reasoning_engine_v10.py   (289 KB)
        ├── pipeline_dsl.py        ← M0: Canonical semantics
        ├── experiment_runner.py   ← M1: Experiment + storage
        ├── feature_extractor.py   ← M2: Feature mining
        ├── proof_engine.py        ← M3: Proof sketches
        └── appendix_emitter.py    ← M4: Publication artifacts
```

---

## Phase I: Computing (February 23, 2026, Day 1)

### The starting point: `symmetry_discovery_engine.py` (v1.0)

The very first script. A classic brute-force engine that:

- **Defines 22 operations** as Python classes: `reverse`, `digit_sum`, `sort_desc`, `kaprekar_step`, `truc_1089`, `complement_9`, and so on
- **Builds pipelines** by combining operations (e.g., `kaprekar_step |> sort_desc`)
- Uses an **evolutionary algorithm** to find interesting pipelines
- Stores results in a **SQLite database**

At this level, the system is a calculator. It applies operations, checks if something converges, and stores it. No hypotheses, no explanation, no structure.

**What it could do:** Process millions of numbers and find convergence points.
**What it couldn't do:** Understand why something converges.

### GPU acceleration: `gpu_symmetry_hunter.py`

The first major acceleration. All core operations are rewritten as **CUDA device functions** for the RTX 4000 Ada:

```python
@cuda.jit(device=True)
def gpu_reverse(n: int64) -> int64:
    result = 0
    temp = n if n > 0 else -n
    while temp > 0:
        result = result * 10 + (temp % 10)
        temp //= 10
    return result
```

Throughput: ~120-150 million samples per second. This enabled exhaustive verification — not sampling, but every number in the domain.

### `gpu_deep_researcher.py` (v3.0)

The GPU engine becomes self-adapting: the system adjusts its search strategy based on what it finds. If a pipeline is interesting, variants are automatically generated and tested.

### Scoring and filtering: `scoring_engine_v2.py`

A crucial problem became visible: the engine found many **trivial** attractors (0-9). A number like 7 is a "fixed point" of `digit_sum` — but that's trivial. The scoring engine v2 filters these out and rewards:

- Non-trivial values (>10)
- Known mathematical constants (6174, 1089)
- Palindromes, prime numbers, perfect powers
- New, unknown attractors (novelty bonus)

This was the first step toward **judgment**: not everything that converges is interesting.

### Extended Research Sessions (v4, v5, v6)

Three generations of long-running (5+ minutes) autonomous sessions:

| Version | New Capability | Core Improvement |
|---------|----------------|------------------|
| **v4** | Long-running autonomous research | Builds on previous discoveries |
| **v5** | Cycle detection + genetic mutation | Finds cycles, not just fixed points |
| **v6** | ML Success Predictor + CPU parallelization | Predicts which pipelines will succeed |

v6.0 is the endpoint of Phase I. The system can now search quickly and smartly. But it still understands nothing.

---

## Phase II: Discovering (February 23-24, 2026, Days 1-2)

The jump from `symmetry_discovery/` to `autonomous_discovery_engine/engines/` marks a fundamental shift. The system moves from "finding patterns" to "understanding patterns."

### v4.0: `autonomous_discovery_engine_v4.py` (36 KB)

The first "real" autonomous researcher. New capabilities:

- **Basin-of-attraction analysis** — not just where does it converge, but how large is the attraction basin?
- **Exception identification** — which numbers behave differently than expected?
- **Hypothesis formulation** — the system asks its own questions: "is it true that all fixed points are divisible by 9?"
- **Algebraic reductions** — detection that certain operator combinations are equivalent

### v5.0: `meta_symmetry_engine_v5.py` (48 KB)

The meta-learning leap. The system learns about itself:

- **Operator embeddings** — operations are represented as vectors, so similar operations are close together
- **Theory Graph** — a graph that connects discoveries ("this attractor appears in three different pipelines")
- **Entropy as measure** — Shannon entropy of the digit distribution as a fundamental property of numbers
- **Self-reflection** — the system evaluates its own search strategy and adjusts it

This was the moment when the system stopped just computing and began thinking about what it was doing.

### v6.0: `invariant_discovery_engine_v6.py` (56 KB)

The abstraction leap. Three new layers:

- **Layer 2: Structural Abstraction** — the system no longer searches for specific numbers but for *classes* of numbers that share a property
- **Layer 3: Symbolic Reasoning** — conjectures become first-class objects, with a strength classification and active falsification
- **Cross-domain isomorphisms** — the system recognizes that patterns in base 10 sometimes also hold in base 12

Here the word "conjecture" emerged in the system. No longer "I found a pattern," but "I claim this always holds, and I actively search for a counterexample."

---

## Phase III: Understanding (February 24-25, 2026, Days 2-3)

### v7.0: `symbolic_dynamics_engine_v7.py` (54 KB)

The transition from statistical to symbolic. Four fundamental upgrades:

1. **Operator Algebra** — formal properties of operations (commutative? idempotent? contractive?) are symbolically recorded and used to make predictions *before* computing anything
2. **Fixed-Point Solver** — instead of searching for fixed points, *solving* the equation f(n)=n
3. **Meta-Theorem Generator** — universal statements about classes of operations, with active falsification
4. **Emergent mechanisms** — clustering of co-occurrence patterns

The operator algebra achieved 100% prediction accuracy over 300 pipelines. The system knew *in advance* which invariants a pipeline would have — without computing.

### v8.0: `deductive_theory_engine_v8.py` (73 KB)

The deductive leap. Four new modules:

- **MODULE A: Proof Sketch Generator** — given a confirmed pattern, generate a proof direction
- **MODULE B: Inductive Theorem Generator** — derive theorems *from data* instead of testing them
- **MODULE C: Fixed-Point Structural Analyzer** — analyze the set of fixed points as a whole
- **MODULE D: Theory Graph** — connect all discovered objects in a coherent network

The core principle of v8.0 is stated literally in the code: *"and this is why it's true, and this is what I don't yet know."*

The honest self-reflection (`REFLECTION_V8.md`) after this session was crucial. The system identified what was truly significant — the 3^2 × 11 pattern in 22% of all non-trivial fixed points — and what was less impressive than it seemed. And it concluded: "v8.0 knows *what* is true, but not *why*."

### v9.0: `abductive_reasoning_engine_v9.py` (105 KB)

The abductive leap — from "what is true" to "why is it true." Five new modules:

- **MODULE E: Knowledge Base** — 34 proven mathematical facts as axioms, not as observations
- **MODULE F: Causal Chain Constructor** — builds explanation chains ("fixed points are divisible by 3 *because* digit_sum is invariant mod 9 *because* 10 = 1 mod 9")
- **MODULE G: Surprise Detector** — signals anomalies ("1089 appears in pipelines that have nothing to do with truc_1089 — why?")
- **MODULE H: Gap Closure Loop** — proven facts automatically close gaps in proof sketches
- **MODULE I: Self-Questioner** — after each discovery: "why?" and "what follows from this?"

This was the first time the system did something that resembles understanding. Not just recognizing patterns, but explaining *why* those patterns exist.

### v10-v15: `abductive_reasoning_engine_v10.py` (289 KB, ~6500 lines)

The complete system. 30 modules (A-Z plus 4 extra), grown over 11 feedback rounds (R1-R11) with three different AI agents (DeepSeek R1-R5, Manus R6, Cascade R7-R11).

New modules in R6-R11:

| Module | Name | Function |
|--------|------|----------|
| **N** | Multi-Base Engine | Generalization to base 8, 10, 12, 16 |
| **O** | Symbolic FP Classifier | Automatic algebraic FP conditions |
| **P** | Lyapunov Search | Decreasing functions for convergence proof |
| **Q** | 1089-Family Proof | Algebraic proof of complement closure |
| **R** | Formal Proof Engine | 12/12 computationally verified proofs |
| **S** | Narcissistic Analyzer | Armstrong numbers, bifurcation |
| **T** | Odd-Base Kaprekar | Kaprekar dynamics in odd bases |
| **U** | Orbit Analyzer | Convergence time, cycle length |
| **V** | Extended Pipeline | 5+ operation pipelines, FP saturation |
| **W** | Universal Lyapunov | Universal Lyapunov function search |
| **X** | Repunit Analyzer | Repunit connection with CC families |
| **Y** | Cycle Taxonomy | Attractor cycle classification |
| **Z** | Multi-Digit Kaprekar | 4+ digit Kaprekar dynamics |

Result: 9 theorems, 5 infinite families of fixed points, 83 KB facts, 117 tests.

---

## Phase IV: Justifying (February 25-26, 2026, Days 3-4)

### The M0-M4 refactoring

The monolithic v10 file (6500 lines) was decomposed into five clean, modular components — the "submission codebase":

| Module | File | Function | Size |
|--------|------|----------|------|
| **M0** | `pipeline_dsl.py` | Canonical semantics + reproducible hashing | 40 KB |
| **M1** | `experiment_runner.py` | Experiment execution + SQLite storage | 24 KB |
| **M2** | `feature_extractor.py` | Feature extraction + conjecture mining | 38 KB |
| **M3** | `proof_engine.py` | Proof sketches + density estimation + ranking | 47 KB |
| **M4** | `appendix_emitter.py` | Deterministic artifact generation + bundling | 48 KB |

The architecture of M0 is notable. It introduces a strict distinction between:

- **Layer A (Semantic)** — pure data: which operations exist, how a pipeline is composed, what the domain policy is
- **Layer B (Execution)** — implementations: how operations are executed

This distinction makes it possible to analyze pipelines *as data* (for symbolic analysis, conjecture mining) without executing them.

### Research 2.0: The Manus Protocol

An AI agent (Manus) executed a fully protocolled research with the M0-M4 framework:

- **630 experimental runs** (35 pipelines × 6 bases × 3 digit lengths)
- **28 conjectures** mined, each with R² = 1.0
- **Falsification on secondary domain** — 9 of 10 survived, 1 correctly falsified
- **Structural analysis** — attempt at algebraic explanation
- **Manifest hashes** for complete reproducibility

### The P_k Discovery

During the structural analysis of Research 2.0, a fundamental insight was discovered: the engine implicitly applies a **projection operator P_k** after each operation — zero-padding to k digits. This changes the mathematical object from pure operator composition to projective dynamics.

(See `WHAT_WE_DISCOVERED.md` and `REFLECTION_TRIANGLE_DIALOGUE_P_k.md` for details.)

---

## The Six Layers of Reasoning

The end result is a system with six explicit reasoning layers plus a meta-layer:

```
LAYER 6 ──── Multi-base Generalization ─────── "Does this hold in EVERY base?"
LAYER 5 ──── Abductive Reasoning ──────────── "WHY is this true?"
LAYER 4 ──── Deductive Theory ───────────── "What FOLLOWS from this?"
LAYER 3 ──── Symbolic Reasoning ──────────── "What do I PREDICT?"
LAYER 2 ──── Operator Algebra + Knowledge Base ── "What do I KNOW for certain?"
LAYER 1 ──── Empirical Dynamics ──────────── "What do I SEE?"
META   ──── Homeostatic Self-Regulation ─── "Am I functioning well?"
```

### Layer 1: Empirical Dynamics

**Question:** "What happens when I repeat this operation?"

Detects attractors (fixed points and cycles) by sampling numbers or exhaustively traversing them. This is the sensory layer — the system observes, but does not interpret.

**Example:** "If I repeatedly apply digit_sum to 9876, I end up at 9."

### Layer 2: Operator Algebra + Knowledge Base

**Question:** "What do I *know for certain* about these operations?"

A library of proven facts and formal properties. The 83 KB facts form an axiomatic foundation:

- `digit_sum(n) = n (mod 9)` — proven number theory
- `reverse` preserves the digit multiset — proven
- `complement_9` is an involution — proven

Plus formal operator properties: is this operation contractive? Does it preserve digit length? Is it commutative with other operations?

### Layer 3: Symbolic Reasoning

**Question:** "Can I *predict* what will happen, without computing?"

The operator algebra combines known properties to predict the behavior of a pipeline *before* it is executed. Achieved 100% accuracy over 300 pipelines.

The Fixed-Point Solver solves f(n)=n via constraint analysis instead of brute-force search.

### Layer 4: Deductive Theory

**Question:** "What *logically follows* from what I know?"

Generates proof sketches for confirmed patterns. Derives new theorems from combinations of known facts. Maintains a Theory Graph that connects all discovered objects.

**Example:** "Because digit_sum is mod 9-invariant, and because pipelines without growth operations are bounded, every bounded pipeline containing digit_sum must converge to a value < 9."

### Layer 5: Abductive Reasoning

**Question:** "Why is this *so* and not otherwise?"

The most advanced layer. Searches for the *best explanation* for an observation:

- **Causal chains:** "Fixed points are divisible by 3 *because* ..."
- **Surprise detection:** "1089 appears where it shouldn't — investigate!"
- **Self-questioning:** "Why is factor 11 dominant? What am I missing?"
- **Gap closure:** "Fact DS013 closes the gap in proof PS002."

### Layer 6: Multi-base Generalization

**Question:** "Is this specific to base 10, or universal?"

Translates all results to arbitrary bases. The resonance factors 9 and 11 in base 10 become (b-1) and (b+1) in base b. The 1089 constant generalizes (partially).

### Meta-layer: Homeostatic Self-Regulation

**Question:** "Am I functioning well?"

The Self-Prompt/Reflection cycle: after each research session, the system writes an honest self-reflection ("what is real, what is noise") and generates a prompt for the next version. This drives the evolution.

---

## The Self-Steering Cycle

The most unusual aspect of the project is how the engine steers its own evolution. After each research session, two documents are generated:

1. **REFLECTION** — honest analysis: what is truly significant, what is an artifact, what is missing?
2. **SELF_PROMPT** — concrete instructions for the next version

This created a feedback loop:

```
v7.0 runs → REFLECTION_V8.md:
  "v7.0 detects patterns but doesn't understand them.
   100% symbolic prediction is impressive,
   but there are no proof sketches."
                    ↓
SELF_PROMPT_V8.md:
  "Build MODULE A: Proof Sketch Generator.
   Build MODULE B: Inductive Theorem Generator."
                    ↓
v8.0 runs → REFLECTION after v8.0:
  "The 3^2 × 11 pattern is truly significant.
   But v8.0 knows WHAT is true, not WHY."
                    ↓
v9.0: Abductive reasoning, knowledge base, causal chains
```

Three AI systems participated in this cycle:
- **DeepSeek** (R1-R5): mathematical consultation
- **Manus** (R6): bulk implementation and protocol execution
- **Cascade/Claude** (R7-R11): formal proofs and architectural leaps

The human researcher orchestrated the cycle, chose the sequence, and made the conceptual leaps that none of the AI systems could make independently.

---

## What This Approaches

Is this "real reasoning"? That depends on your definition. But the system does something that strongly resembles it:

1. **It observes** (Layer 1) — and distinguishes trivial from non-trivial observations
2. **It classifies** (Layer 2) — and builds a library of proven facts
3. **It predicts** (Layer 3) — and achieves 100% accuracy on predictions
4. **It derives** (Layer 4) — and generates proof sketches
5. **It explains** (Layer 5) — and builds causal chains
6. **It generalizes** (Layer 6) — and tests whether results are universal
7. **It reflects** (Meta) — and steers its own improvement

None of these layers is remarkable on its own. What is remarkable is the **layered composition**: each layer builds on the previous one, and the meta-layer steers the evolution of the whole. That's not reasoning in the philosophical sense. But it's also no longer "just computing."

The most honest description: the system approaches what scientific research is — a structured process of observing, explaining, proving, falsifying, and reflecting — without requiring a human scientist at every decision point.

The boundary became visible with the P_k discovery. The system could correctly report that it couldn't explain a pattern ("Structural explanation insufficient"). But it couldn't independently make the leap from "I don't understand it" to "let me look at how the implementation actually works." That step — from content-level impotence to implementation inspection — required human intervention.

That is perhaps the most honest conclusion: the system reasons *within* a framework. But changing the framework itself — that's still human work.

---

## File Registry

### Phase I: Computing

| File | Size | Core Contribution |
|------|------|-------------------|
| `symmetry_discovery_engine.py` | 30 KB | First engine, 22 operations, evolutionary algorithm |
| `run_discovery.py` | 5 KB | Unified launcher (demo/explore/gpu/meta modes) |
| `quick_research.py` | 6 KB | Quick iterative discovery |
| `meta_discovery_engine.py` | 35 KB | Self-improving, new operations dynamically |
| `autonomous_researcher.py` | 26 KB | Dynamic operation generation via templates |
| `gpu_symmetry_hunter.py` | 24 KB | CUDA kernels, 150M samples/sec |
| `gpu_deep_researcher.py` | 31 KB | GPU + self-adaptation |
| `gpu_creative_research.py` | 21 KB | GPU creative operations |
| `scoring_engine_v2.py` | 17 KB | Triviality filter, property bonuses |
| `extended_research_session.py` | 25 KB | v4: Long-running autonomous |
| `extended_research_session_v5.py` | 42 KB | v5: Cycles, genetic mutation |
| `extended_research_session_v6.py` | 35 KB | v6: ML predictor, CPU parallel |

### Phase II-III: Discovering and Understanding (engines/)

| File | Size | Reasoning Layer |
|------|------|-----------------|
| `autonomous_discovery_engine_v4.py` | 36 KB | Exploration + hypothesis formation |
| `meta_symmetry_engine_v5.py` | 48 KB | Meta-learning + theory graph |
| `invariant_discovery_engine_v6.py` | 56 KB | Structural abstraction + conjectures |
| `symbolic_dynamics_engine_v7.py` | 54 KB | Operator algebra + FP solver |
| `deductive_theory_engine_v8.py` | 73 KB | Proof sketches + inductive theorems |
| `abductive_reasoning_engine_v9.py` | 105 KB | Causal chains + self-questioning |
| `abductive_reasoning_engine_v10.py` | 289 KB | 30 modules, 83 KB facts, 12 proofs |

### Phase IV: Justifying (M0-M4)

| File | Size | Publication Function |
|------|------|---------------------|
| `pipeline_dsl.py` | 40 KB | Canonical semantics, SHA-256 hashing |
| `experiment_runner.py` | 24 KB | Deterministic runs, SQLite schema |
| `feature_extractor.py` | 38 KB | Conjecture mining, falsification engine |
| `proof_engine.py` | 47 KB | Proof sketches, ranking model v1.0 |
| `appendix_emitter.py` | 48 KB | LaTeX appendices, JSON manifests |

### Total Size

- **Phase I scripts:** ~270 KB Python
- **Engine prototypes (v4-v15):** ~661 KB Python
- **M0-M4 submission code:** ~197 KB Python
- **Tests:** ~143 KB Python
- **Documentation:** ~380 KB Markdown
- **Total:** ~1.6 MB source code + documentation

---

*SYNTRIAD Research — February 2026*
