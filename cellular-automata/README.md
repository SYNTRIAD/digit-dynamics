# Cellular Automata — Fixed-Point Sequences

This directory contains research into fixed-point counts of Wolfram elementary
cellular automata (1D, radius-1, cyclic lattice) and their connection to
known integer sequences in the OEIS.

## Key Results

| CA Rule | Sequence | Name | Growth |
|---------|----------|------|--------|
| Rule 4  | A000204  | Lucas numbers | phi = 1.6180... |
| Rule 76 | A001644  | Tribonacci numbers | tau = 1.8392... |

The fixed-point count FP(Rule R, k) equals the trace of the k-th power of
the 4x4 de Bruijn transfer matrix for rule R. The dominant eigenvalue of
this matrix determines the asymptotic growth class. Both identities were
established by exhaustive enumeration (n=1..14) and confirmed via the
transfer-matrix trace for arbitrary n.

## OEIS status: assessed, not submitted

The connection above is mathematically correct and reproducible. However,
the planned OEIS *annotation* proposals for A000204 and A001644 were
**assessed and not pursued**. Both sequences already exist in the OEIS as
mature, well-referenced entries; a CA-fixed-point annotation was judged to
add limited value and was not submitted. The artifacts under `oeis/` are
retained as verified internal documentation of the identity, not as live
submission proposals. See the AXIOM OEIS tracker for the full disposition
rationale across all candidate sequences.

## Structure

```
cellular-automata/
├── verify_submissions.py       <- standalone reproducibility runner
├── oeis/
│   ├── A000204/                <- Lucas-number identity (internal record)
│   └── A001644/                <- Tribonacci identity (internal record)
└── runs/
    ├── RES-001_axiom_5683fce6/ <- original session (superseded)
    └── RES-003_axiom_baf19226/ <- corrected rerun (basis of the identity)
```

## Quick Start

```bash
python verify_submissions.py          # full verification + eigenvalue analysis
python verify_submissions.py --quick  # n=1..14 verification only
```

No dependencies beyond Python 3.8+ stdlib. `numpy` optional for eigenvalue output.
Exit code 0 = all verifications passed.

## Connection to digit-dynamics

The digit-dynamics project studies fixed points and cycles of digit-based
operators (Kaprekar, digit sums, etc.). Cellular automata present the same
mathematical structure — fixed points of a local operator on a cyclic space —
at a different scale. The transfer matrix method used here is directly
analogous to the transition matrix approach in symbolic dynamics.

## Research Sessions

| Session | ID | Status | Notes |
|---------|----|--------|-------|
| RES-001 | axiom_research_5683fce6 | Superseded | Sparse k-range, OEIS checker crash |
| RES-003 | axiom_research_baf19226 | **Definitive** | Continuous k=1..14, all fixes |

Certificate (RES-003): `ef1f5c359b05d86a942d3c229dfd97b5bcc8bf94dfc95de89a395be8da15b134`
