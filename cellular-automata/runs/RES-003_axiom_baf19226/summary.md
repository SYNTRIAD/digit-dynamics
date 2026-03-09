# RES-003 Rerun — Deep CA Research Session Summary

## Session Overview

| Field | Value |
|-------|-------|
| **Session ID** | `axiom_research_baf19226` |
| **Date** | 2026-03-01 |
| **Type** | Rerun of `axiom_research_5683fce6` (RES-001) |
| **Domain** | Cellular Automata (Wolfram Elementary CA) |
| **Rules Tested** | 0, 4, 30, 54, 76, 90, 110, 150, 255 |
| **k-range** | 1–14 (continuous, no gaps) |
| **Experiments** | 135 (126 CA + 9 OEIS checks) |
| **Claims** | 9 |
| **Certificate** | HMAC-verified VALID |

## Key Findings

### Rule 4 → A000204 (Lucas numbers) — EXACT match, confidence 1.0

```
FP(Rule 4, k) for k=1..14:
1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843
```

Growth ratio → phi = 1.618048 (golden ratio).
Transfer matrix T = [[1,1],[1,0]], char. poly x^2 - x - 1.

### Rule 76 → A001644 (Tribonacci numbers) — confirmed EXACT (offset k-1)

```
FP(Rule 76, k) for k=1..14:
1, 3, 7, 11, 21, 39, 71, 131, 241, 443, 815, 1499, 2757, 5071
```

Growth ratio → tau = 1.839269 (Tribonacci constant).
Transfer matrix 4x4, char. poly x(x^3 - x^2 - x - 1).

## Taxonomy of CA Fixed-Point Growth

| Class | Rules | Growth | OEIS |
|-------|-------|--------|------|
| Constant | 0, 54, 110, 255 | O(1) | — |
| Periodic | 30, 90, 150 | O(1) | A010684, A210954, A010694 |
| Fibonacci-class | 4 | O(phi^k) | **A000204** |
| Tribonacci-class | 76 | O(tau^k) | **A001644** |

## Why This Run Supersedes RES-001

| Fix | RES-001 Problem | This Run |
|-----|-----------------|----------|
| OEIS Checker | crash on `.terms` | fixed, 5/5 detected |
| k-range | sparse gaps | continuous 1..14 |
| Exponential detection | tolerance failures | last-3-ratio, tol=0.05 |

---
*AXIOM v2.1.0 — Certificate: ef1f5c359b05d86a942d3c229dfd97b5bcc8bf94dfc95de89a395be8da15b134*
