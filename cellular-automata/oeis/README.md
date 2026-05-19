# OEIS — Cellular Automata Identities (internal records)

Two CA-fixed-point identities for existing OEIS sequences, based on the
discovery that fixed-point counts of Wolfram elementary CA rules match
known integer sequences.

## Identities

| File | Sequence | Claim |
|------|----------|-------|
| A000204/A000204_SUBMISSION.txt | Lucas numbers | FP(Rule 4, n) = L(n) |
| A001644/A001644_SUBMISSION.txt | Tribonacci numbers | FP(Rule 76, k) = a(k-1) |

## Status: assessed, not submitted

These were prepared as annotation proposals (not new sequences) and
verified by exhaustive enumeration for n=1..14, confirmed via the
transfer-matrix trace for arbitrary n. The mathematics is correct.

The annotation submissions themselves were **assessed and not pursued**:
A000204 and A001644 are already mature, well-referenced OEIS entries, and
a CA-fixed-point annotation was judged to add limited value. The files in
this directory are retained as verified internal documentation of the
identities, not as live submission drafts.

See `../verify_submissions.py` to reproduce all results independently.
