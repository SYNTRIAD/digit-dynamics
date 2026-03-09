# OEIS Submissions

Two annotation proposals for existing OEIS sequences, based on the
discovery that fixed-point counts of Wolfram elementary CA rules match
known integer sequences.

## Submissions

| File | Sequence | Claim |
|------|----------|-------|
| A000204/A000204_SUBMISSION.txt | Lucas numbers | FP(Rule 4, n) = L(n) |
| A001644/A001644_SUBMISSION.txt | Tribonacci numbers | FP(Rule 76, k) = a(k-1) |

Both submissions are annotation proposals (not new sequences).
Verified by exhaustive enumeration for n=1..14 and confirmed via
transfer matrix trace for arbitrary n.

See `../verify_submissions.py` to reproduce all results independently.
