#!/usr/bin/env python3
# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research — MIT License
"""
verify_submissions.py
======================
Standalone reproducibility runner for OEIS submissions:
  - A000204: Lucas numbers = fixed points of Rule 4 on cyclic lattice of n cells
  - A001644: Tribonacci numbers = fixed points of Rule 76 on cyclic lattice of n+1 cells

No external dependencies beyond Python 3.8+ stdlib.
numpy is optional (enables eigenvalue analysis; not required for verification).

Usage:
    python verify_submissions.py          # full verification + analysis
    python verify_submissions.py --quick  # verification only (n=1..14)

Expected output: all MATCH columns show OK, exit code 0.
"""
from __future__ import annotations

import sys
import math
from typing import Dict, List, Tuple, Optional


# =============================================================================
# 1. Wolfram elementary CA -- exhaustive fixed-point enumeration
# =============================================================================

def _rule_table(rule_number: int) -> Dict[Tuple[int, int, int], int]:
    """Build lookup table for a Wolfram elementary CA rule."""
    return {
        ((i >> 2) & 1, (i >> 1) & 1, i & 1): (rule_number >> i) & 1
        for i in range(8)
    }


def ca_step(config: Tuple[int, ...], rule_number: int) -> Tuple[int, ...]:
    """Apply one step of rule_number on a cyclic lattice."""
    table = _rule_table(rule_number)
    k = len(config)
    return tuple(
        table[(config[(i - 1) % k], config[i], config[(i + 1) % k])]
        for i in range(k)
    )


def count_fixed_points(n: int, rule_number: int) -> int:
    """Exhaustive count of fixed points for lattice size n under rule_number."""
    count = 0
    for bits in range(1 << n):
        config = tuple((bits >> (n - 1 - i)) & 1 for i in range(n))
        if ca_step(config, rule_number) == config:
            count += 1
    return count


# =============================================================================
# 2. Transfer matrix -- exact FP counts via Tr(M^k), O(log k)
# =============================================================================

def _mat_mul(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    n = len(A)
    return [
        [sum(A[i][p] * B[p][j] for p in range(n)) for j in range(n)]
        for i in range(n)
    ]


def _mat_pow(M: List[List[int]], k: int) -> List[List[int]]:
    n = len(M)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in M]
    while k > 0:
        if k & 1:
            result = _mat_mul(result, base)
        base = _mat_mul(base, base)
        k >>= 1
    return result


def build_transfer_matrix(rule_number: int) -> List[List[int]]:
    """Build the 4x4 de Bruijn transfer matrix for fixed-point enumeration."""
    table = _rule_table(rule_number)
    M = [[0] * 4 for _ in range(4)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                if table[(a, b, c)] == b:
                    M[a * 2 + b][b * 2 + c] += 1
    return M


def fp_via_trace(rule_number: int, k: int) -> int:
    """Compute |FP(rule, k)| = Tr(M^k) via the transfer matrix."""
    M = build_transfer_matrix(rule_number)
    Mk = _mat_pow(M, k)
    return sum(Mk[i][i] for i in range(4))


# =============================================================================
# 3. Reference sequences
# =============================================================================

def lucas(n: int) -> int:
    """A000204: Lucas numbers L(n), n >= 1. L(1)=1, L(2)=3, ..."""
    if n == 1: return 1
    if n == 2: return 3
    a, b = 1, 3
    for _ in range(n - 2):
        a, b = b, a + b
    return b


def tribonacci_A001644(n: int) -> int:
    """A001644: a(0)=1, a(1)=3, a(2)=7, a(n)=a(n-1)+a(n-2)+a(n-3)."""
    if n == 0: return 1
    if n == 1: return 3
    if n == 2: return 7
    a, b, c = 1, 3, 7
    for _ in range(n - 2):
        a, b, c = b, c, a + b + c
    return c


# =============================================================================
# 4. Verification
# =============================================================================

def verify_A000204(n_max: int = 14) -> bool:
    """Verify: FP(Rule 4, n) == L(n) for n = 1..n_max"""
    print("=" * 62)
    print("A000204 -- Lucas numbers = Rule 4 fixed points")
    print("=" * 62)
    print(f"{'n':>3}  {'FP exhaustive':>14}  {'FP trace(T^n)':>14}  {'L(n)':>8}  {'match':>5}")
    print("-" * 62)
    all_ok = True
    for n in range(1, n_max + 1):
        fp_ex = count_fixed_points(n, rule_number=4)
        fp_tm = fp_via_trace(rule_number=4, k=n)
        ln    = lucas(n)
        ok    = (fp_ex == fp_tm == ln)
        all_ok = all_ok and ok
        print(f"{n:>3}  {fp_ex:>14}  {fp_tm:>14}  {ln:>8}  {'OK' if ok else 'FAIL':>5}")
    print("-" * 62)
    print(f"Result: {'ALL MATCH' if all_ok else 'FAILURES DETECTED'}")
    print()
    return all_ok


def verify_A001644(n_max: int = 14) -> bool:
    """Verify: FP(Rule 76, k) == a(k-1) for k = 1..n_max (A001644, offset 0)."""
    print("=" * 62)
    print("A001644 -- Tribonacci numbers = Rule 76 fixed points")
    print("  FP(k) = a(k-1),  lattice size = k,  OEIS offset: a(0)=1")
    print("=" * 62)
    print(f"{'k':>3}  {'FP exhaustive':>14}  {'FP trace(T^k)':>14}  {'a(k-1)':>8}  {'match':>5}")
    print("-" * 62)
    all_ok = True
    for k in range(1, n_max + 1):
        fp_ex = count_fixed_points(k, rule_number=76)
        fp_tm = fp_via_trace(rule_number=76, k=k)
        ak1   = tribonacci_A001644(k - 1)
        ok    = (fp_ex == fp_tm == ak1)
        all_ok = all_ok and ok
        print(f"{k:>3}  {fp_ex:>14}  {fp_tm:>14}  {ak1:>8}  {'OK' if ok else 'FAIL':>5}")
    print("-" * 62)
    print(f"Result: {'ALL MATCH' if all_ok else 'FAILURES DETECTED'}")
    print()
    return all_ok


# =============================================================================
# 5. Optional: eigenvalue analysis (requires numpy)
# =============================================================================

def eigenvalue_analysis():
    try:
        import numpy as np
    except ImportError:
        print("numpy not available -- skipping eigenvalue analysis")
        print("Install with: pip install numpy")
        return

    print("=" * 62)
    print("Eigenvalue analysis (requires numpy)")
    print("=" * 62)
    for rule, label, expected, name in [
        (4,  "A000204 / Rule 4",  (1 + math.sqrt(5)) / 2, "golden ratio phi"),
        (76, "A001644 / Rule 76", 1.8392867552,            "tribonacci constant tau"),
    ]:
        M = np.array(build_transfer_matrix(rule), dtype=float)
        eigenvalues = np.linalg.eigvals(M)
        dominant = max((e.real for e in eigenvalues if abs(e.imag) < 1e-8), default=0.0)
        char_poly = np.round(np.poly(M)).astype(int).tolist()
        print(f"\nRule {rule} ({label})")
        print(f"  Transfer matrix: {[list(row) for row in M.astype(int)]}")
        print(f"  Char. polynomial: {char_poly}")
        print(f"  Dominant eigenvalue: {dominant:.15f}")
        print(f"  Expected ({name}):  {expected:.15f}")
        print(f"  Match within 1e-6:   {abs(dominant - expected) < 1e-6}")
    print()


# =============================================================================
# 6. Large-k demonstration
# =============================================================================

def large_k_demo():
    print("=" * 62)
    print("Large-k demonstration (exact integer arithmetic, no numpy)")
    print("=" * 62)
    for rule, label in [(4, "Rule 4 / Lucas"), (76, "Rule 76 / Tribonacci")]:
        print(f"\n{label}:")
        for k in [50, 100, 200]:
            print(f"  |FP({k:3d})| = {fp_via_trace(rule, k)}")
    print()


# =============================================================================
# 7. Main
# =============================================================================

def main():
    quick = "--quick" in sys.argv
    ok_204  = verify_A000204(n_max=14)
    ok_1644 = verify_A001644(n_max=14)
    if not quick:
        eigenvalue_analysis()
        large_k_demo()
    overall = ok_204 and ok_1644
    print("=" * 62)
    print(f"OVERALL: {'ALL VERIFICATIONS PASSED' if overall else 'FAILURES DETECTED'}")
    print("=" * 62)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
