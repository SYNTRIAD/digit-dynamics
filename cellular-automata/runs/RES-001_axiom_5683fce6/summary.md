# RES-001 — Original CA Research Session (axiom_research_5683fce6)

**Status:** Superseded by RES-003 (axiom_research_baf19226)

## Session Overview

| Field | Value |
|-------|-------|
| **Session ID** | `axiom_research_5683fce6` |
| **Date** | 2026-03-01 20:53 UTC |
| **Rules Tested** | 0, 4, 30, 54, 76, 90, 110, 150, 255 |
| **k-range** | sparse: 3,4,5,6,7,8,10,12 |
| **Experiments** | 61 |
| **Claims** | 9 |
| **Certificate** | HMAC-verified VALID |

## Key Findings (manual identification)

| Rule | Sequence | Match |
|------|----------|-------|
| Rule 4 | A000204 (Lucas) | manually identified |
| Rule 76 | A001644 (Tribonacci) | manually identified |

## Known Issues (fixed in RES-003)

| Issue | Effect |
|-------|--------|
| Sparse k-range with gaps (no k=1,2,9,11,13,14) | Growth ratios distorted — `likely_exponential: false` for Rules 4 and 76 |
| OEISChecker crash on `.terms` attribute | `oeis_match: null` for all rules in oeis_results.json |
| Exponential detection tolerance too tight | All sequences classified as non-exponential |

Despite these bugs, the raw `fp_counts` in analysis.json and findings.json
are correct — the enumeration itself was sound. OEIS matches were identified
manually by pattern recognition after the session.

## Certificate
`2595ca5444982c669e94742491a3507df7aa93f6880915faae0329dcbfc0e269`
